"""Course entry point: `python main.py` reads data/raw/test_images and writes submission.csv.

Same as `python scripts/predict.py` (which takes --images, --weights, --out, --device).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from predict import main

if __name__ == "__main__":
    main()
