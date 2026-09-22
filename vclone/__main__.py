import sys

from .cli import main

if __name__ == "__main__":  # the voice step's spawned process (LatentSync videos) must not re-run the CLI
    sys.exit(main())
