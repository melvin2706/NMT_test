"""KamerNMT as a translation service for external applications (JSON in, JSON out).

Start it:
    kamernmt-server                          # http://127.0.0.1:8000
    kamernmt-server --host 0.0.0.0 --port 8080 --model models/kamernmt-small-v1-int8
    python server.py ...                     # same thing, from the project folder

Request  — POST /translate   (Content-Type: application/json)
    {
      "source_language": "fr",                 required
      "target_language": "dua",                required
      "content": <text | list | JSON object>,  required: the data to translate
      "skip_keys": ["id", "url"],              optional: values under these keys stay as they are
      "only_keys": ["title", "body"]           optional: translate only values under these keys
    }

Response — 200
    {
      "source_language": "fr",
      "target_language": "dua",
      "content": <same structure as the request's content, values translated>,
      "route": ["fr", "dua"],                  languages the text went through (French = pivot)
      "model": "kamernmt-small-v1-int8"
    }

    POST /translate?download=1  returns only the translated content, as a JSON file
    (Content-Disposition: attachment; filename="translation.<target>.json").

Only values are translated, never keys. Numbers, booleans and null are unchanged.
Placeholders ({name}, %s, <b>…</b>, URLs, e-mails) are kept.

Other endpoints:
    GET /health      {"status": "ok", "model": …, "pivot": "fr", "device": "cuda (+cpu fallback)" | "cpu", "languages": [...]}
    GET /languages   {"languages": [...], "trained_directions": [...], "pivot": "fr"}

Errors: 400 (missing field, unknown language, invalid JSON), 404, 413 (request too big),
500 — always as JSON: {"error": "…"}.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REQUIRED = ("source_language", "target_language", "content")
MAX_BODY = 20 * 1024 * 1024  # 20 MB


class RequestError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _key_set(value, name: str) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, list) or not all(isinstance(k, str) for k in value):
        raise RequestError(f"'{name}' must be a list of key names")
    return set(value)


def handle_request(engine, req, lock: threading.Lock | None = None) -> dict:
    """Validate a translation request and translate it. Used by the HTTP server
    and by `kamernmt --request file.json`."""
    if not isinstance(req, dict):
        raise RequestError("the request must be a JSON object")
    missing = [k for k in REQUIRED if k not in req]
    if missing:
        raise RequestError(f"missing required field(s): {', '.join(missing)}")
    src, tgt, content = req["source_language"], req["target_language"], req["content"]
    known = engine.languages
    for field, code in (("source_language", src), ("target_language", tgt)):
        if not isinstance(code, str) or code not in known:
            raise RequestError(f"unknown {field} {code!r}; available: {', '.join(known)}")
    if content is None:
        raise RequestError("'content' is empty")
    skip = _key_set(req.get("skip_keys"), "skip_keys")
    only = _key_set(req.get("only_keys"), "only_keys") or None

    def run():
        if isinstance(content, str):
            return engine.translate_text(content, src, tgt)
        # lists and objects: same structure back, values translated, keys untouched
        return engine.translate_document(content, src, tgt, only_keys=only, skip_keys=skip)

    if lock:
        with lock:
            translated = run()
    else:
        translated = run()
    hops = engine.route(src, tgt)
    return {
        "source_language": src,
        "target_language": tgt,
        "content": translated,
        "route": [src] + [b for _, b in hops] if hops else [src],
        "model": Path(engine.model_path).name,
    }


def make_handler(engine, cors_origin: str | None, quiet: bool):
    lock = threading.Lock()  # one translation at a time: the model is shared

    class Handler(BaseHTTPRequestHandler):
        server_version = "KamerNMT"

        def _send(self, status: int, payload, filename: str | None = None) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            if cors_origin:
                self.send_header("Access-Control-Allow-Origin", cors_origin)
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):  # browser pre-flight (CORS)
            self.send_response(204)
            if cors_origin:
                self.send_header("Access-Control-Allow-Origin", cors_origin)
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            path = urlparse(self.path).path.rstrip("/")
            if path in ("", "/health"):
                self._send(200, {"status": "ok", "model": Path(engine.model_path).name,
                                 "pivot": engine.pivot, "device": engine.device, "languages": engine.languages})
            elif path == "/languages":
                self._send(200, {"languages": engine.languages, "pivot": engine.pivot,
                                 "trained_directions": engine.card.get("trained_directions", [])})
            else:
                self._send(404, {"error": f"unknown path {path}; use POST /translate"})

        def do_POST(self):
            url = urlparse(self.path)
            if url.path.rstrip("/") != "/translate":
                return self._send(404, {"error": f"unknown path {url.path}; use POST /translate"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    raise RequestError("empty request body")
                if length > MAX_BODY:
                    raise RequestError(f"request too big (max {MAX_BODY // 1024 // 1024} MB)", 413)
                try:
                    req = json.loads(self.rfile.read(length).decode("utf-8-sig"))
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    raise RequestError(f"invalid JSON: {e}")
                result = handle_request(engine, req, lock)
            except RequestError as e:
                return self._send(e.status, {"error": str(e)})
            except Exception as e:  # never crash the server on one bad request
                return self._send(500, {"error": f"translation failed: {e}"})
            if parse_qs(url.query).get("download", ["0"])[0] in ("1", "true", "yes"):
                self._send(200, result["content"], filename=f"translation.{result['target_language']}.json")
            else:
                self._send(200, result)

        def log_message(self, fmt, *args):
            if not quiet:
                sys.stderr.write(f"[kamernmt] {self.address_string()} {fmt % args}\n")

    return Handler


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="kamernmt-server", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 = this computer only; 0.0.0.0 = reachable from the network")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", help="model folder (default: the most recent in models/)")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--no-pivot", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto = GPU if available, with the CPU as backup (default)")
    ap.add_argument("--cors", metavar="ORIGIN",
                    help="allow browser apps from this origin to call the API, e.g. https://app.example.cm or *")
    ap.add_argument("--quiet", action="store_true", help="do not log each request")
    a = ap.parse_args(argv)

    from .engine import AUTO, TranslationEngine

    try:
        engine = TranslationEngine(a.model, pivot=None if a.no_pivot else AUTO, beam=a.beam, device=a.device)
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(f"kamernmt-server: {e}")
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(engine, a.cors, a.quiet))
    print(f"KamerNMT translation service on http://{a.host}:{a.port}", flush=True)
    print(f"model: {engine.model_path}   pivot: {engine.pivot}   device: {engine.device}")
    print(f"languages: {', '.join(engine.languages)}")
    print("POST /translate  {source_language, target_language, content}   —   Ctrl+C to stop", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
