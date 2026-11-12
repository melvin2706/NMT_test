"""Start the KamerNMT translation service from the project folder (no installation needed):

    python server.py                       # http://127.0.0.1:8000
    python server.py --host 0.0.0.0 --port 8080

Same as the `kamernmt-server` command (see `python server.py --help`).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kamernmt.server import main  # noqa: E402

if __name__ == "__main__":
    main()
