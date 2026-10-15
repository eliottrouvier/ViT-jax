"""
ViT-JAX — Téléchargement & Préparation de STL-10 en RAM
Récupère les images 96x96 non étiquetées et sauvegarde data/stl10_packed.pt
pour des passes avant/arrière ultra-rapides sur Apple Silicon MPS.
"""

import os
import sys
import ssl
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets

ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = DATA_DIR / "stl10_packed.pt"


def prepare_stl10(num_train: int = 20000, num_test: int = 1000):
    if OUT_FILE.exists():
        print(f"✅ {OUT_FILE} existe déjà. Chargement des dimensions...")
        d = torch.load(OUT_FILE, weights_only=True)
        print(f"  Train: {d['train_imgs'].shape}, Test: {d['test_imgs'].shape}")
        return

    print("📥 Téléchargement de STL-10 (split='unlabeled', 96x96 pixels)...")
    ds_unlabeled = datasets.STL10(root=str(DATA_DIR), split="unlabeled", download=True)

    print(f"📦 Emballage de {num_train} images en RAM...")
    # STL-10 stocke les images en shape (N, 3, 96, 96) uint8
    train_slice = ds_unlabeled.data[:num_train]
    train_imgs = torch.from_numpy(train_slice)

    print(f"📥 Récupération du split test ({num_test} images pour visualisations)...")
    ds_test = datasets.STL10(root=str(DATA_DIR), split="test", download=True)
    test_slice = ds_test.data[:num_test]
    test_imgs = torch.from_numpy(test_slice)

    print(f"💾 Sauvegarde dans {OUT_FILE}...")
    torch.save({
        "train_imgs": train_imgs,
        "test_imgs": test_imgs,
    }, OUT_FILE)

    file_size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
    print(f"🎉 Succès ! STL-10 prêt en mémoire : {OUT_FILE} ({file_size_mb:.1f} Mo)")


if __name__ == "__main__":
    prepare_stl10()
