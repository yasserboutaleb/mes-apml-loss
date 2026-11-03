"""Small deterministic CPU-only example; synthetic data, no benchmark claims."""
from __future__ import annotations

import torch
from torch import nn
from mes_apml import apml_loss, mes_loss, combined_loss


def run() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(1)
    class_count = 4
    y = torch.arange(class_count).repeat_interleave(30)
    prototypes = torch.eye(class_count)
    x = prototypes[y] + 0.25 * torch.randn(len(y), class_count)

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(class_count, 3)
            self.proxies = nn.Parameter(0.2 * torch.randn(class_count, 3))

        def forward(self, data):
            return self.encoder(data)

    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    for step in range(101):
        z = model(x)
        objective = combined_loss(z, y, model.proxies, alpha=1.0, beta=0.05)
        if step % 25 == 0:
            print(f"step={step:3d} APML={apml_loss(z, y, model.proxies).item():.5f} "
                  f"MES={mes_loss(model.proxies).item():.5f}")
        if step == 100:
            break
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    print("This is a fixed synthetic CPU demonstration, not an accuracy benchmark.")


if __name__ == "__main__":
    run()
