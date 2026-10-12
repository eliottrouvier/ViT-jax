"""
ViT-JAX — Phase 1.8 : Entraînement Supervisé Ultra-Rapide sur CIFAR-10 / CIFAR-100
Références :
- Dosovitskiy et al., ICLR 2021 (papiers/vit.pdf)
- Touvron et al. (DeiT), ICML 2021 (papiers/deit.pdf)

Optimisations haute performance :
- Pré-chargement 100% en RAM du dataset (0 lecture disque pendant les époques).
- Augmentations vectorisées PyTorch natives (Flip horizontal + Random Crop padding réfléchi).
- Redimensionnement bicubique vectorisé sur GPU (MPS / CUDA).
- Optimiseur AdamW avec Cosine Annealing et Label Smoothing (0.1).
- Sauvegarde de la courbe d'apprentissage haute résolution dans outputs/ et des poids .pt.
"""

import argparse
import os
import sys
import time
import ssl
from pathlib import Path
import matplotlib.pyplot as plt

# Contournement standard sur macOS pour les certificats SSL
ssl._create_default_https_context = ssl._create_unverified_context

# Chemins de base et ajout au sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from pytorch.models import vit_base_patch16_224, vit_tiny_patch16_224

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class FastInMemoryDataset(Dataset):
    """Dataset haute performance stocké intégralement en RAM."""
    def __init__(self, imgs, targets, mean, std, is_train=True):
        self.imgs = imgs.clone()        # (N, 3, 32, 32) uint8
        self.targets = targets.clone()  # (N,) int64
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.is_train = is_train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].float() / 255.0

        if self.is_train:
            # Random Horizontal Flip vectorisé
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])
            # Random Crop avec padding=4 réfléchi
            pad = 4
            img_padded = F.pad(img, (pad, pad, pad, pad), mode="reflect")
            h_off = torch.randint(0, 2 * pad + 1, (1,)).item()
            w_off = torch.randint(0, 2 * pad + 1, (1,)).item()
            img = img_padded[:, h_off : h_off + 32, w_off : w_off + 32]

        # Normalisation
        img = (img - self.mean) / self.std
        return img, self.targets[idx]


def get_fast_cifar10_loaders(batch_size: int = 128, num_workers: int = 0):
    """Charge ou génère le dataset packed en RAM."""
    packed_file = DATA_DIR / "cifar10_packed.pt"
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    if not packed_file.exists():
        print("Pré-chargement des images CIFAR-10 depuis data/cifar10...")
        train_ds = datasets.ImageFolder(DATA_DIR / "cifar10" / "train", transform=transforms.ToTensor())
        loader = DataLoader(train_ds, batch_size=2500, shuffle=False, num_workers=2)
        imgs, tgts = [], []
        for x, y in loader:
            imgs.append((x * 255).to(torch.uint8))
            tgts.append(y)
        train_imgs = torch.cat(imgs, dim=0)
        train_targets = torch.cat(tgts, dim=0)

        test_ds = datasets.ImageFolder(DATA_DIR / "cifar10" / "test", transform=transforms.ToTensor())
        loader_t = DataLoader(test_ds, batch_size=2500, shuffle=False, num_workers=2)
        imgs_t, tgts_t = [], []
        for x, y in loader_t:
            imgs_t.append((x * 255).to(torch.uint8))
            tgts_t.append(y)
        test_imgs = torch.cat(imgs_t, dim=0)
        test_targets = torch.cat(tgts_t, dim=0)

        torch.save(
            {"train_imgs": train_imgs, "train_targets": train_targets,
             "test_imgs": test_imgs, "test_targets": test_targets},
            packed_file
        )

    print("Chargement immédiat du tenseur RAM CIFAR-10...")
    data = torch.load(packed_file, weights_only=True)
    train_dataset = FastInMemoryDataset(data["train_imgs"], data["train_targets"], mean, std, is_train=True)
    test_dataset = FastInMemoryDataset(data["test_imgs"], data["test_targets"], mean, std, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, 10


def train_one_epoch(model, loader, criterion, optimizer, device, img_size=64, max_batches=None):
    """Entraîne le modèle sur une époque avec interpolation bicubique GPU vectorisée."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    start_time = time.time()

    for idx, (images, targets) in enumerate(loader):
        if max_batches and idx >= max_batches:
            break

        images = images.to(device)
        targets = targets.to(device)

        # Redimensionnement GPU vectorisé instantané
        if images.shape[-1] != img_size:
            images = F.interpolate(images, size=(img_size, img_size), mode="bicubic", align_corners=False)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(loader):
            print(f"  Batch [{idx+1}/{len(loader)}] | Loss: {loss.item():.4f} | Acc: {100.0 * correct / total:.2f}%")

    elapsed = time.time() - start_time
    epoch_loss = total_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, elapsed


@torch.no_grad()
def evaluate(model, loader, criterion, device, img_size=64, max_batches=None):
    """Évalue le modèle sur l'ensemble de test."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for idx, (images, targets) in enumerate(loader):
        if max_batches and idx >= max_batches:
            break

        images = images.to(device)
        targets = targets.to(device)

        if images.shape[-1] != img_size:
            images = F.interpolate(images, size=(img_size, img_size), mode="bicubic", align_corners=False)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * targets.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


def plot_learning_curves(history, out_path, title="Convergence ViT"):
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

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Courbes de convergence sauvegardées : {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Entraînement supervisé ultra-rapide ViT")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=128, help="Taille de batch")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate initial")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight Decay AdamW")
    parser.add_argument("--model", type=str, default="vit_tiny", choices=["vit_tiny", "vit_base"],
                        help="vit_tiny (5.5M rapide) ou vit_base (85.8M canonique)")
    parser.add_argument("--img-size", type=int, default=64, help="Résolution d'entrée du ViT (64, 112 ou 224)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limite optionnelle de batchs par époque pour un test rapide")
    args = parser.parse_args()

    # Détection automatique du meilleur périphérique matériel
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("  ViT-JAX — Phase 1.8 : Entraînement Supervisé sur CIFAR-10 (Haute Vitesse)")
    print("=" * 70)
    print(f"• Périphérique d'accélération : {device}")
    print(f"• Modèle sélectionné         : {args.model}")
    print(f"• Résolution d'image         : {args.img_size}x{args.img_size} (Interpolation Bicubique 2D)")
    print(f"• Époques demandées          : {args.epochs}")
    print(f"• Batch Size                 : {args.batch_size}")
    print(f"• Learning Rate (AdamW)      : {args.lr}")

    train_loader, test_loader, num_classes = get_fast_cifar10_loaders(
        batch_size=args.batch_size
    )

    # Instanciation du modèle avec interpolation dynamique de pos_embed
    if args.model == "vit_base":
        model = vit_base_patch16_224(num_classes=num_classes)
    else:
        model = vit_tiny_patch16_224(num_classes=num_classes)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"• Paramètres du modèle       : {total_params / 1e6:.2f} M")
    print(f"• Classes cibles             : {num_classes}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine Annealing avec Warmup sur 1 époque
    warmup_epochs = 1
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        else:
            import math
            progress = float(epoch - warmup_epochs) / float(max(1, args.epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\n🚀 Démarrage de l'entraînement...")
    for epoch in range(1, args.epochs + 1):
        lr_current = optimizer.param_groups[0]["lr"]
        print(f"\n--- Époque [{epoch}/{args.epochs}] (lr={lr_current:.6f}) ---")
        train_loss, train_acc, elapsed = train_one_epoch(
            model, train_loader, criterion, optimizer, device, img_size=args.img_size, max_batches=args.max_batches
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device, img_size=args.img_size, max_batches=args.max_batches
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  👉 Bilan Époque {epoch} ({elapsed:.1f}s) :")
        print(f"     Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"     Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.2f}%")

    # Sauvegarde des poids
    ckpt_path = OUTPUTS_DIR / f"{args.model}_cifar10.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n💾 Poids sauvegardés : {ckpt_path}")

    # Sauvegarde des courbes
    plot_path = OUTPUTS_DIR / "train_cifar10_convergence.png"
    title = f"Convergence {args.model} sur CIFAR-10 ({args.epochs} Époques, {args.img_size}x{args.img_size})"
    plot_learning_curves(history, plot_path, title=title)

    print(f"\n✅ Entraînement terminé : Précision finale de {history['val_acc'][-1]:.2f}% !")


if __name__ == "__main__":
    main()
