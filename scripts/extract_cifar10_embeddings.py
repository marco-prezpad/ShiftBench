#!/usr/bin/env python3
"""
extract_cifar10_embeddings.py

Extract embeddings from CIFAR-10 using a pre-trained ResNet18.
Saves train/test embeddings and labels to embeddings/cifar10/.

Author: Marco Pérez Padilla
Date:   13-08-2026
"""
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

OUT_DIR = Path("embeddings/cifar10")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model.to(device)
model.eval()

embedder_layers = list(model.children())[:-1]
embedder = torch.nn.Sequential(*embedder_layers).to(device)


def extract_embeddings(loader):
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            features = embedder(images).squeeze()
            embeddings.append(features.cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.vstack(embeddings), np.concatenate(labels)


print("Extracting embeddings...")
train_embeddings, train_labels = extract_embeddings(train_loader)
np.save(OUT_DIR / "train_embeddings.npy", train_embeddings)
np.save(OUT_DIR / "train_labels.npy", train_labels)
print(f"Train embeddings saved: {train_embeddings.shape}")

print("Extracting test embeddings...")
test_embeddings, test_labels = extract_embeddings(test_loader)
np.save(OUT_DIR / "test_embeddings.npy", test_embeddings)
np.save(OUT_DIR / "test_labels.npy", test_labels)
print(f"Test embeddings saved: {test_embeddings.shape}")
