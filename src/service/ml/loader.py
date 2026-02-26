
from dataclasses import dataclass
from pathlib import Path

import torch

from .model import MLP
from ..core.config import DEVICE, DEFAULT_MEAN, DEFAULT_STD

@dataclass
class ModelBundle:
    model: MLP
    mean: float
    std: float
    model_path: Path

def load_model(model_path: Path) -> ModelBundle:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    bundle = torch.load(model_path, map_location=DEVICE)

    model = MLP().to(DEVICE)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    mean = float(bundle.get("mean", DEFAULT_MEAN))
    std = float(bundle.get("std", DEFAULT_STD))

    return ModelBundle(model=model, mean=mean, std=std, model_path=model_path)