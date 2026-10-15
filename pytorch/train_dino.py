"""
ViT-JAX — Phase 2B : Entraînement Auto-Supervisé DINO (Self-Distillation)
Référence : Mathilde Caron et al., ICCV 2021 (papiers/dino.pdf)
"Emerging Properties in Self-Supervised Vision Transformers"

Pré-entraînement 100% auto-supervisé (SANS AUCUNE ÉTIQUETTE) :
- Pipeline Multi-Crop (2 vues globales + 4 vues locales).
- Couple Élève (Student) / Professeur (Teacher).
- Tête de projection DINO avec prototypes (K=1024).
- Centrage & Sharpening (évite l'effondrement représentationnel).
- Mise à jour EMA du Professeur.
- Sauvegarde des poids dans outputs/dino_vit_tiny.pt.
"""

import argparse
import os
import sys
import time
import math
import ssl
from pathlib import Path
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from pytorch.self_supervised import (
    DataAugmentationDINO,
    DINOHead,
    DINOLoss,
    DINOModel,
    update_teacher_ema,
    dino_vit_tiny,
    dino_vit_base,
)

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class FastRAMDINODataset(Dataset):
    """Dataset rapide en mémoire appliquant le pipeline Multi-Crop DINO."""
    def __init__(self, imgs: torch.Tensor, transform: DataAugmentationDINO):
        self.imgs = imgs.clone()
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Conversion [0, 255] uint8 -> [0.0, 1.0] float32
        img = self.imgs[idx].float() / 255.0
        return self.transform(img)


def dino_collate_fn(batch):
    """
    Collate spécifique DINO : regroupe les N cultures en listes de tenseurs batchés.
    Retourne : List[torch.Tensor] de longueur ncrops.
    """
    ncrops = len(batch[0])
    return [torch.stack([item[i] for item in batch]) for i in range(ncrops)]


def get_data_loader(
    dataset: str = "stl10",
    batch_size: int = 128,
    global_size: int = 96,
    local_size: int = 48,
    local_crops_number: int = 4,
) -> DataLoader:
    """Prépare le DataLoader Multi-Crop."""
    transform = DataAugmentationDINO(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.1, 0.4),
        local_crops_number=local_crops_number,
        global_size=global_size,
        local_size=local_size,
    )

    stl10_packed = DATA_DIR / "stl10_packed.pt"
    cifar10_packed = DATA_DIR / "cifar10_packed.pt"

    if dataset.lower() == "stl10" and stl10_packed.exists():
        print(f"Chargement immédiat de STL-10 depuis {stl10_packed}...")
        data = torch.load(stl10_packed, weights_only=True)
        dataset_obj = FastRAMDINODataset(data["train_imgs"], transform=transform)
    elif cifar10_packed.exists():
        print("Chargement immédiat des images depuis data/cifar10_packed.pt...")
        data = torch.load(cifar10_packed, weights_only=True)
        dataset_obj = FastRAMDINODataset(data["train_imgs"], transform=transform)
    else:
        print("Chargement depuis torchvision...")
        to_tensor = transforms.ToTensor()
        if dataset.lower() == "stl10":
            raw_ds = datasets.STL10(root=str(DATA_DIR), split="unlabeled", download=True)
            class STL10DINODataset(Dataset):
                def __init__(self, ds, t):
                    self.ds = ds
                    self.t = t
                def __len__(self): return len(self.ds)
                def __getitem__(self, i):
                    img_pil, _ = self.ds[i]
                    return self.t(to_tensor(img_pil))
            dataset_obj = STL10DINODataset(raw_ds, transform)
        else:
            folder = DATA_DIR / "cifar10" / "train"
            raw_ds = datasets.ImageFolder(folder)
            class DiskDINODataset(Dataset):
                def __init__(self, ds, t):
                    self.ds = ds
                    self.t = t
                def __len__(self): return len(self.ds)
                def __getitem__(self, i):
                    img_pil, _ = self.ds[i]
                    return self.t(to_tensor(img_pil))
            dataset_obj = DiskDINODataset(raw_ds, transform)

    loader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=dino_collate_fn,
    )
    return loader


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    dino_loss: DINOLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    momentum_schedule: list,
    max_batches: int = None,
):
    """Entraîne une époque DINO complète."""
    student.train()
    teacher.train()
    total_loss, total_batches = 0.0, 0
    start_time = time.time()

    for idx, crops in enumerate(loader):
        if max_batches and idx >= max_batches:
            break

        # Envoi de toutes les cultures sur le device
        crops = [c.to(device) for c in crops]
        n_global = 2
        global_crops = crops[:n_global]
        local_crops = crops[n_global:]

        # 1. Forward du Teacher (uniquement sur les 2 vues globales, sans gradient)
        teacher_inputs = torch.cat(global_crops, dim=0)
        with torch.no_grad():
            teacher_output = teacher(teacher_inputs)

        # 2. Forward du Student (sur toutes les vues : globales + locales)
        # Note : On passe séparément les globales et locales si leurs résolutions diffèrent
        student_global = student(torch.cat(global_crops, dim=0))
        if len(local_crops) > 0:
            student_local = student(torch.cat(local_crops, dim=0))
            student_output = torch.cat([student_global, student_local], dim=0)
        else:
            student_output = student_global

        # 3. Calcul de la perte Cross-Entropy avec Centering & Sharpening
        loss = dino_loss(student_output, teacher_output, epoch=epoch)

        # 4. Rétropropagation et descente de gradient sur l'Élève
        optimizer.zero_grad()
        loss.backward()
        # Normalisation des gradients
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
        # On annule les gradients sur la magnitude des prototypes normalisés
        if hasattr(student.head.last_layer, "weight_g"):
            student.head.last_layer.weight_g.grad = None
        optimizer.step()

        # 5. Mise à jour EMA du Professeur
        m = momentum_schedule[epoch] if epoch < len(momentum_schedule) else 0.996
        update_teacher_ema(student, teacher, momentum=m)

        total_loss += loss.item()
        total_batches += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(loader):
            print(f"  Batch [{idx+1}/{len(loader)}] | DINO Self-Distillation Loss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    epoch_loss = total_loss / max(1, total_batches)
    return epoch_loss, elapsed


def plot_loss_curve(losses, out_path):
    """Sauvegarde le graphique de décroissance de la perte DINO."""
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, "o-", color="#8b5cf6", lw=2, label="DINO Cross-Entropy Loss (Centering+Sharpening)")
    plt.title("Convergence du Pré-entraînement Auto-Supervisé DINO", fontsize=12, fontweight="bold")
    plt.xlabel("Époque")
    plt.ylabel("Perte Cross-Entropy Auto-Supervisée")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Courbe d'apprentissage DINO sauvegardée : {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Pré-entraînement auto-supervisé DINO")
    parser.add_argument("--dataset", type=str, default="stl10", choices=["stl10", "cifar10"],
                        help="Dataset pour le pré-entraînement")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'époques de pré-entraînement")
    parser.add_argument("--batch-size", type=int, default=64, help="Taille de batch")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate initial")
    parser.add_argument("--weight-decay", type=float, default=0.04, help="Weight decay AdamW")
    parser.add_argument("--out-dim", type=int, default=1024, help="Dimension de l'espace de prototypes")
    parser.add_argument("--global-size", type=int, default=96, help="Résolution des vues globales (ex: 96 ou 224)")
    parser.add_argument("--local-size", type=int, default=48, help="Résolution des vues locales (ex: 48 ou 96)")
    parser.add_argument("--patch-size", type=int, default=8, help="Taille des patchs (ex: 8 ou 16)")
    parser.add_argument("--local-crops-number", type=int, default=2, help="Nombre de vues locales")
    parser.add_argument("--model", type=str, default="dino_vit_tiny", choices=["dino_vit_tiny", "dino_vit_base"],
                        help="dino_vit_tiny (rapide) ou dino_vit_base (86M)")
    parser.add_argument("--max-batches", type=int, default=None, help="Limite optionnelle de batchs par époque")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("  ViT-JAX — Phase 2B : Pré-entraînement Auto-Supervisé DINO")
    print("=" * 70)
    print(f"• Périphérique        : {device}")
    print(f"• Dataset             : {args.dataset.upper()}")
    print(f"• Modèle              : {args.model}")
    print(f"• Taille des patchs   : {args.patch_size}x{args.patch_size}")
    print(f"• Prototypes (K)      : {args.out_dim}")
    print(f"• Vues globales       : 2 x ({args.global_size}x{args.global_size})")
    print(f"• Vues locales        : {args.local_crops_number} x ({args.local_size}x{args.local_size})")
    print(f"• Époques demandées   : {args.epochs}")
    print(f"• Batch Size          : {args.batch_size}")
    print(f"• Learning Rate       : {args.lr}")

    loader = get_data_loader(
        dataset=args.dataset,
        batch_size=args.batch_size,
        global_size=args.global_size,
        local_size=args.local_size,
        local_crops_number=args.local_crops_number,
    )

    ncrops = 2 + args.local_crops_number
    dino_loss = DINOLoss(
        out_dim=args.out_dim,
        ncrops=ncrops,
        student_temp=0.1,
        teacher_temp=0.04,
        center_momentum=0.9,
    ).to(device)

    # Création du couple Student / Teacher
    if args.model == "dino_vit_base":
        student, teacher = dino_vit_base(out_dim=args.out_dim, img_size=args.global_size, patch_size=args.patch_size)
    else:
        student, teacher = dino_vit_tiny(out_dim=args.out_dim, img_size=args.global_size, patch_size=args.patch_size)

    student = student.to(device)
    teacher = teacher.to(device)

    total_params = sum(p.numel() for p in student.parameters())
    print(f"• Paramètres Student  : {total_params / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Momentum schedule du Teacher : monte doucement de 0.996 à 1.0
    momentum_schedule = [0.996 + 0.004 * (epoch / max(1, args.epochs)) for epoch in range(args.epochs)]

    losses = []
    print("\n🚀 Démarrage du pré-entraînement auto-supervisé DINO...")
    for epoch in range(args.epochs):
        print(f"\n--- Époque [{epoch + 1}/{args.epochs}] (lr={optimizer.param_groups[0]['lr']:.6f}, EMA m={momentum_schedule[epoch]:.4f}) ---")
        epoch_loss, elapsed = train_one_epoch(
            student=student,
            teacher=teacher,
            dino_loss=dino_loss,
            loader=loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            momentum_schedule=momentum_schedule,
            max_batches=args.max_batches,
        )
        scheduler.step()
        losses.append(epoch_loss)
        print(f"  👉 Bilan Époque {epoch + 1} ({elapsed:.1f}s) | Loss DINO: {epoch_loss:.4f}")

    # Sauvegarde des checkpoints (on sauvegarde les poids du Teacher car ils sont plus stables)
    ckpt_path = OUTPUTS_DIR / f"{args.model}.pt"
    torch.save({
        "teacher": teacher.state_dict(),
        "student": student.state_dict(),
        "dino_loss": dino_loss.state_dict(),
    }, ckpt_path)
    print(f"\n💾 Poids DINO (Teacher & Student) sauvegardés : {ckpt_path}")

    # Sauvegarde du graphique de perte
    plot_path = OUTPUTS_DIR / "train_dino_convergence.png"
    plot_loss_curve(losses, plot_path)

    print("\n✅ Pré-entraînement DINO achevé avec succès !")


if __name__ == "__main__":
    main()
