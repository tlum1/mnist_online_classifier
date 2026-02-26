# save_mnist_to_folders.py
from pathlib import Path
from torchvision import datasets
import numpy as np
from PIL import Image
from tqdm import tqdm

def save_split(ds, out_dir: Path, prefix: str = ""):
    out_dir.mkdir(parents=True, exist_ok=True)

    for c in range(10):
        (out_dir / str(c)).mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(ds)), desc=f"Saving {out_dir.name}"):
        img, y = ds[i]  # PIL, int
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img), mode="L")

        fname = f"{prefix}{i:07d}.png"
        img.save(out_dir / str(y) / fname)

def main():
    root = Path("src/nn/mnist_folders")
    data_root = Path("mnist_raw")

    train_ds = datasets.MNIST(root=str(data_root), train=True, download=True, transform=None)
    test_ds  = datasets.MNIST(root=str(data_root), train=False, download=True, transform=None)

    save_split(train_ds, root / "train", prefix="tr_")
    save_split(test_ds,  root / "test",  prefix="te_")

    print(f"Done. Saved to: {root.resolve()}")

if __name__ == "__main__":
    main()