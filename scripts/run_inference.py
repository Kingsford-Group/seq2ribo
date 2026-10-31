#!/usr/bin/env python3
"""Command-line inference; same as the `seq2ribo` command installed by `pip install -e .`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # this repository's seq2ribo first
from seq2ribo.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
