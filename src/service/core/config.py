from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # .../src
WEB_DIR = BASE_DIR / "web"
MODEL_PATH = BASE_DIR / "models" / "mnist_mlp.pt"

DEVICE = "cpu"
DEFAULT_MEAN = 0.1307
DEFAULT_STD = 0.3081