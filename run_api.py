#!/usr/bin/env python
"""Wrapper script to start the API from the repository root."""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "pricing"

if not PROJECT_DIR.exists():
    raise SystemExit(f"Project directory not found: {PROJECT_DIR}")

os.chdir(PROJECT_DIR)

# Re-execute the nested run_api.py from the correct working directory.
args = [sys.executable, str(PROJECT_DIR / "run_api.py")] + sys.argv[1:]
os.execv(sys.executable, args)
