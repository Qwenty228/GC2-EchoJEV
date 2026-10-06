"""Command-line pipelines that train and score detectors and write runs/<run id>/."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

# One pipeline command: takes the parsed CLI arguments, returns whatever it likes.
Command = Callable[[argparse.Namespace], Any]
