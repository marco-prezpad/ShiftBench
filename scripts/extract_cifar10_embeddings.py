#!/usr/bin/env python3
"""
extract_cifar10_embeddings.py

Extract embeddings from CIFAR-10 using a pre-trained ResNet18.
Saves train/test embeddings and labels to embeddings/cifar10/.

Author: Marco Pérez Padilla
Date:   13-08-2026
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

OUT_DIR = Path("embeddings/cifar10")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

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

modules = list(model.children())[:-1]
embedder = torch.nn.Sequential(*modules).to(device)

def extract_embeddings(loader):
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            feats = embedder(images).squeeze()  
            embeddings.append(feats.cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.vstack(embeddings), np.concatenate(labels)

print("Extrayendo embeddings de entrenamiento...")
train_emb, train_lbl = extract_embeddings(train_loader)
np.save(OUT_DIR / "train_embeddings.npy", train_emb)
np.save(OUT_DIR / "train_labels.npy", train_lbl)
print(f"Train embeddings guardados: {train_emb.shape}")

print("Extrayendo embeddings de test...")
test_emb, test_lbl = extract_embeddings(test_loader)
np.save(OUT_DIR / "test_embeddings.npy", test_emb)
np.save(OUT_DIR / "test_labels.npy", test_lbl)
print(f"Test embeddings guardados: {test_emb.shape}")