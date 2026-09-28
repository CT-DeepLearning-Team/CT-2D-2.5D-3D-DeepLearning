import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.common.cache import build
build(force="--force" in sys.argv)
