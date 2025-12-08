"""Allow running enyal as: python -m enyal"""

import sys

from enyal.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
