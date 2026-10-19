"""Thin CLI entry points for existing data preparation and verification scripts."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]


def prepare():
    runpy.run_path(str(ROOT / 'scripts' / 'prepare_data.py'), run_name='__main__')


def verify():
    sys.argv = ['data-verify', '--verify', *sys.argv[1:]]
    prepare()
