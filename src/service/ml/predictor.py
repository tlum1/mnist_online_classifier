import torch
from PIL import Image

from ..core.config import DEVICE
from .loader import ModelBundle
from .preprocess import preprocess_mnist_like

@torch.no_grad()
def predict_image(img: Image.Image, bundle: ModelBundle) -> tuple[int, list[float]]:
    x = preprocess_mnist_like(img, mean=bundle.mean, std=bundle.std).to(DEVICE)
    logits = bundle.model(x)
    probs_t = torch.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs_t).item())
    probs = probs_t.cpu().tolist()
    return pred, probs