#!/usr/bin/env python3
"""Run from this folder, or invoke by absolute path."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
from codetest.cli import main

if __name__ == "__main__":
    main()
