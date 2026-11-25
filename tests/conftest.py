"""
conftest.py
pytest loads this file before any test. It puts src/ on the import path so that
scripts loaded by file path (tests/test_pipeline.py) can import config.py and
conformal.py, exactly as they do when run with `py src/xx_name.py`.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
