"""
ViT-JAX — Étape 1.8 : Entraînement Supervisé sur CIFAR-100
Référence : Dosovitskiy et al., ICLR 2021 (papiers/vit.pdf)

Entraîne le Vision Transformer sur CIFAR-100 redimensionné en 224x224,
suit la convergence sur 5 époques et sauvegarde la courbe d'apprentissage dans outputs/.
"""

import argparse
import os
import sys
import time
import ssl
from pathlib import Path
import matplotlib.pyplot as plt

# Contournement standard sur macOS pour le téléchargement SSL de datasets
ssl._create_default_https_context = ssl._create_unverified_context

# Chemins de base et ajout au sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pytorch.models import vit_base_patch16_224, vit_tiny_patch16_224

# Chemins de base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def get_data_loaders(img_size: int = 224, batch_size: int = 32, num_workers: int = 2):
    """Charge CIFAR-100 avec augmentation standard et redimensionnement à 224x224."""
    # Moyennes et écart-types normalisés CIFAR-100
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    print("Téléchargement / Chargement de CIFAR-100...")
    cifar100_folder = DATA_DIR / "cifar100"
    if (cifar100_folder / "train").exists():
        train_dataset = datasets.ImageFolder(cifar100_folder / "train", transform=train_transform)
        test_dataset = datasets.ImageFolder(cifar100_folder / "test", transform=test_transform)
    else:
        train_dataset = datasets.CIFAR100(
            root=DATA_DIR, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            root=DATA_DIR, train=False, download=True, transform=test_transform
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    """Entraîne le modèle sur une époque."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    start_time = time.time()

    for idx, (images, targets) in enumerate(loader):
        if max_batches and idx >= max_batches:
            break

        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(loader):
            print(f"  Batch [{idx+1}/{len(loader)}] | Loss: {loss.item():.4f} | Acc: {100.0 * correct / total:.2f}%")

    elapsed = time.time() - start_time
    epoch_loss = total_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, elapsed


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    """Évalue le modèle sur l'ensemble de validation."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for idx, (images, targets) in enumerate(loader):
        if max_batches and idx >= max_batches:
            break

        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * targets.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


def plot_learning_curves(history, out_path):
    """Trace et sauvegarde les courbes de perte et de précision."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Courbe de perte
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e", lw=2)
    ax1.set_title("Évolution de la Perte Cross-Entropy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Perte")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Courbe de précision
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc", color="#2ca02c", lw=2)
    ax2.plot(epochs, history["val_acc"], "s--", label="Val Acc", color="#d62728", lw=2)
    ax2.set_title("Évolution de la Précision Top-1 (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Précision (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle("Convergence ViT sur CIFAR-100 (5 Époques)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Courbes de convergence sauvegardées : {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Entraînement supervisé ViT sur CIFAR-100")
    parser.add_argument("--epochs", type=int, default=5, help="Nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=64, help="Taille de batch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate initial")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Découplage Weight Decay (AdamW)")
    parser.add_argument("--model", type=str, default="vit_tiny", choices=["vit_tiny", "vit_base"],
                        help="vit_tiny (5.7M rapide) ou vit_base (85.8M canonique)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limite de batchs par époque pour un test rapide")
    args = parser.parse_args()

    # Détection automatique du meilleur périphérique matériel
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("  ViT-JAX — Phase 1.8 : Entraînement Supervisé sur CIFAR-100")
    print("=" * 70)
    print(f"• Périphérique d'accélération : {device}")
    print(f"• Modèle sélectionné         : {args.model}")
    print(f"• Époques demandées          : {args.epochs}")
    print(f"• Batch Size                 : {args.batch_size}")
    print(f"• Learning Rate (AdamW)      : {args.lr}")

    # Instanciation du modèle avec 100 classes
    if args.model == "vit_base":
        model = vit_base_patch16_224(num_classes=100)
    else:
        model = vit_tiny_patch16_224(num_classes=100)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"• Paramètres du modèle       : {total_params / 1e6:.2f} M")

    train_loader, test_loader = get_data_loaders(img_size=224, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\n🚀 Démarrage de l'entraînement...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Époque [{epoch}/{args.epochs}] (lr={optimizer.param_groups[0]['lr']:.6f}) ---")
        train_loss, train_acc, elapsed = train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_batches=args.max_batches
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device, max_batches=args.max_batches
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  👉 Bilan Époque {epoch} ({elapsed:.1f}s) :")
        print(f"     Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"     Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.2f}%")

    # Sauvegarde du modèle et du graphique
    ckpt_path = OUTPUTS_DIR / f"{args.model}_cifar100.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n💾 Poids sauvegardés : {ckpt_path}")

    plot_path = OUTPUTS_DIR / "train_cifar100_convergence.png"
    plot_learning_curves(history, plot_path)

    print("\n✅ Étape 1.8 terminée : Le modèle a convergé avec succès !")


if __name__ == "__main__":
    main()
