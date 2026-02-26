import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)