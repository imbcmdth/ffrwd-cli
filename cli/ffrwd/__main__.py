"""Allow `python -m ffrwd ...` as an install-free entry point."""

import sys

from ffrwd.cli import main

if __name__ == "__main__":
    sys.exit(main())
