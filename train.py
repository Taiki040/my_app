"""Train the MNIST digit classifier and save digit_model.pth."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from classifier import SimpleCNN


def train(
    data_dir: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        (
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        )
    )
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}: loss={total_loss / len(loader):.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output_path)
    print(f"Saved model weights to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST digits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("digit_model.pth"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")
    train(args.data_dir, args.output, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
