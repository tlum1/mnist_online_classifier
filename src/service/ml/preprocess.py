import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter

def preprocess_mnist_like(img: Image.Image, mean: float, std: float) -> torch.Tensor:

    img = img.convert("L")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    arr = np.array(img, dtype=np.uint8)

    thr = 40
    mask = arr > thr
    if not mask.any():
        x = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        return x

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    cropped = Image.fromarray(arr[y0:y1, x0:x1], mode="L")

    w, h = cropped.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))
    else:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))

    cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(cropped, ((28 - new_w) // 2, (28 - new_h) // 2))

    a = np.array(canvas, dtype=np.float32)
    m = a.sum()
    if m > 0:
        ys2, xs2 = np.indices(a.shape)
        cy = (ys2 * a).sum() / m
        cx = (xs2 * a).sum() / m

        shift_x = int(round(13.5 - cx))
        shift_y = int(round(13.5 - cy))
        canvas = ImageChops.offset(canvas, shift_x, shift_y)

    x = torch.tensor(np.array(canvas, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    x = (x - mean) / std
    return x