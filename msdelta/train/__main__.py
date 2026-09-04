"""Allow `python -m msdelta.train`, which the Polaris launcher uses."""

import sys

from msdelta.train.cli import main

if __name__ == "__main__":
    sys.exit(main())
