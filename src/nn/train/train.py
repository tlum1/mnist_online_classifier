# train_from_folders.py
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),         
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    loss_sum, acc_sum, n = 0.0, 0.0, 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        bs = x.size(0)
        loss_sum += loss.item() * bs
        acc_sum  += accuracy(logits, y) * bs
        n += bs

    return loss_sum / n, acc_sum / n

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0

    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        bs = x.size(0)
        loss_sum += loss.item() * bs
        acc_sum  += accuracy(logits, y) * bs
        n += bs

    return loss_sum / n, acc_sum / n

def main():
    data_root = Path("src/nn/mnist_folders")
    model_out_path = Path("src/models")
    train_dir = data_root / "train"
    test_dir  = data_root / "test"
    assert train_dir.exists() and test_dir.exists(), "Не нашёл mnist_folders/train и mnist_folders/test"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),          
        transforms.ToTensor(),                                
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),   
    ])

    train_ds = datasets.ImageFolder(root=str(train_dir), transform=tfm)
    test_ds  = datasets.ImageFolder(root=str(test_dir),  transform=tfm)

    print("class_to_idx:", train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2, pin_memory=(device=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=(device=="cuda"))

    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = 0.0
    epochs = 10

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        te_loss, te_acc = eval_one_epoch(model, test_loader, loss_fn, device)

        print(f"epoch {epoch:02d}/{epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.4f}")

        if te_acc > best_acc:
            best_acc = te_acc
            ckpt = {
                "state_dict": model.state_dict(),
                "mean": MNIST_MEAN,
                "std": MNIST_STD,
                "arch": "MLP(784-256-128-10)"
            }
            torch.save(ckpt, model_out_path / "mnist_mlp.pt")
            print(f"  saved mnist_mlp.pt (best_acc={best_acc:.4f})")

    print("done. best_acc:", best_acc)

if __name__ == "__main__":
    main()