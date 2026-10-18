"""Memungkinkan ``python -m imgenhance``."""

import sys

from .cli import main

sys.exit(main())
