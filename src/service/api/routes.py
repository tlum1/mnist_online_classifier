import io
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Request
from PIL import Image

from ..core.config import MODEL_PATH
from ..ml.loader import ModelBundle
from ..ml.predictor import predict_image

router = APIRouter()

def get_bundle(request: Request) -> ModelBundle:
    bundle: Optional[ModelBundle] = request.app.state.model_bundle
    if bundle is None:
        raise RuntimeError("Model is not loaded")
    return bundle

@router.get("/health")
def health(request: Request):
    bundle: Optional[ModelBundle] = request.app.state.model_bundle
    return {
        "ok": True,
        "model_loaded": bundle is not None,
        "model_path": str(MODEL_PATH),
    }

@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    bundle = get_bundle(request)

    content = await file.read()
    img = Image.open(io.BytesIO(content))

    pred, probs = predict_image(img, bundle)

    return {
        "pred": pred,
        "probs": probs,
        "debug": {"size_in": img.size, "size_proc": [28, 28]},
    }