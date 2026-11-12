"""Translate from the project folder without installing anything:

    python translate.py                                    # interactive mode
    python translate.py --tgt dua "Je vais au marché."     # phrases
    python translate.py --tgt dua --input doc.json         # JSON document

Same as the `kamernmt` command (see `python translate.py --help`).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kamernmt.translate_cli import main  # noqa: E402

if __name__ == "__main__":
    main()
