import sys
from pathlib import Path

# Ensure repository root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.inspect_model_inputs import inspect_model


if __name__ == "__main__":
    inspect_model(str(ROOT / "model" / "model.pkl"))
