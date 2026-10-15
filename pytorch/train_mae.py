"""
ViT-JAX — Phase 2A : Entraînement Auto-Supervisé MAE (Masked Autoencoder)
Référence : Kaiming He et al., CVPR 2022 (papiers/mae.pdf)
"Masked Autoencoders Are Scalable Vision Learners"

Pré-entraînement 100% auto-supervisé (SANS AUCUNE ÉTIQUETTE) :
- Masque 75% des patchs aléatoirement.
- L'encodeur n'analyse que les 25% de patchs visibles.
- Le décodeur reconstruit les pixels des 75% masqués.
- Support natif haute résolution STL-10 (96x96, patch_size=8 -> 144 tokens).
- Sauvegarde des poids pré-entraînés dans outputs/mae_vit_tiny.pt.
"""

import argparse
import os
import sys
import time
import ssl
from pathlib import Path
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from pytorch.self_supervised import (
    MaskedAutoencoderViT,
    mae_vit_base_patch16_224,
    mae_vit_tiny_patch16_224,
)

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class FastRAMDataset(Dataset):
    """Dataset auto-supervisé stocké en RAM (sans labels nécessaires)."""
    def __init__(self, imgs: torch.Tensor):
        self.imgs = imgs.clone()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Conversion [0, 255] uint8 -> [0.0, 1.0] float32
        img = self.imgs[idx].float() / 255.0
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[2])
        return img


def get_data_loader(dataset_name: str = "stl10", batch_size: int = 128) -> DataLoader:
    """Charge STL-10 ou CIFAR-10 depuis le tenseur RAM."""
    stl10_packed = DATA_DIR / "stl10_packed.pt"
    cifar_packed = DATA_DIR / "cifar10_packed.pt"

    if dataset_name == "stl10" and stl10_packed.exists():
        print(f"Chargement immédiat de STL-10 depuis {stl10_packed}...")
        data = torch.load(stl10_packed, weights_only=True)
        dataset = FastRAMDataset(data["train_imgs"])
    elif cifar_packed.exists():
        print(f"Chargement immédiat de CIFAR-10 depuis {cifar_packed}...")
        data = torch.load(cifar_packed, weights_only=True)
        dataset = FastRAMDataset(data["train_imgs"])
    else:
        print("Génération de données synthétiques de secours...")
        dataset = FastRAMDataset(torch.randint(0, 256, (1000, 3, 96, 96), dtype=torch.uint8))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


def train_one_epoch(model, loader, optimizer, device, img_size=96, mask_ratio=0.75, max_batches=None):
    """Entraîne MAE sur une époque complète."""
    model.train()
    total_loss, total_batches = 0.0, 0
    start_time = time.time()

    for idx, images in enumerate(loader):
        if max_batches and idx >= max_batches:
            break

        images = images.to(device)
        # Redimensionnement GPU vectorisé si nécessaire
        if images.shape[-1] != img_size:
            images = F.interpolate(images, size=(img_size, img_size), mode="bicubic", align_corners=False)

        optimizer.zero_grad()
        loss, pred, mask = model(images, mask_ratio=mask_ratio)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if (idx + 1) % 40 == 0 or (idx + 1) == len(loader):
            print(f"  Batch [{idx+1}/{len(loader)}] | MAE Reconstruction Loss: {loss.item():.4f}")

    elapsed = time.time() - start_time
    epoch_loss = total_loss / max(1, total_batches)
    return epoch_loss, elapsed


def plot_loss_curve(losses, out_path):
    """Sauvegarde le graphique de décroissance de la perte de reconstruction."""
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, "o-", color="#e11d48", lw=2, label="MAE MSE Loss (75% Masqué - STL-10)")
    plt.title("Convergence du Pré-entraînement Auto-Supervisé MAE (STL-10 96x96)", fontsize=12, fontweight="bold")
    plt.xlabel("Époque")
    plt.ylabel("Perte MSE (Pixels Normalisés)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Courbe d'apprentissage MAE sauvegardée : {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Pré-entraînement auto-supervisé MAE")
    parser.add_argument("--dataset", type=str, default="stl10", choices=["stl10", "cifar10"],
                        help="Dataset source (défaut: stl10 96x96)")
    parser.add_argument("--epochs", type=int, default=15, help="Nombre d'époques de pré-entraînement")
    parser.add_argument("--batch-size", type=int, default=128, help="Taille de batch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate initial")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay AdamW")
    parser.add_argument("--mask-ratio", type=float, default=0.75, help="Ratio de masquage (defaut 0.75 / 75 pourcent)")
    parser.add_argument("--model", type=str, default="mae_vit_tiny", choices=["mae_vit_tiny", "mae_vit_base"],
                        help="mae_vit_tiny (rapide) ou mae_vit_base (86M)")
    parser.add_argument("--img-size", type=int, default=96, help="Résolution de l'image (96 pour STL-10)")
    parser.add_argument("--patch-size", type=int, default=8, help="Taille des patchs (8 pour 144 tokens fins)")
    parser.add_argument("--max-batches", type=int, default=None, help="Limite optionnelle de batchs par époque")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_patches = (args.img_size // args.patch_size) ** 2
    print("=" * 70)
    print("  ViT-JAX — Phase 2A : Pré-entraînement Auto-Supervisé MAE")
    print("=" * 70)
    print(f"• Périphérique        : {device}")
    print(f"• Dataset             : {args.dataset.upper()}")
    print(f"• Modèle              : {args.model}")
    print(f"• Résolution d'image  : {args.img_size}x{args.img_size}")
    print(f"• Taille des patchs   : {args.patch_size}x{args.patch_size} ({num_patches} tokens au total)")
    print(f"• Ratio de masquage   : {args.mask_ratio * 100:.0f}% ({int(num_patches * (1 - args.mask_ratio))} visibles, {int(num_patches * args.mask_ratio)} masqués)")
    print(f"• Époques demandées   : {args.epochs}")
    print(f"• Batch Size          : {args.batch_size}")
    print(f"• Learning Rate       : {args.lr}")

    loader = get_data_loader(dataset_name=args.dataset, batch_size=args.batch_size)

    # Instanciation du modèle avec la résolution et la granularité de patchs spécifiées
    if args.model == "mae_vit_base":
        model = MaskedAutoencoderViT(
            img_size=args.img_size, patch_size=args.patch_size, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16, norm_pix_loss=True,
        )
    else:
        model = MaskedAutoencoderViT(
            img_size=args.img_size, patch_size=args.patch_size, embed_dim=192, depth=12, num_heads=3,
            decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4, norm_pix_loss=True,
        )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"• Paramètres totaux   : {total_params / 1e6:.2f} M (Encodeur + Décodeur)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    losses = []
    print("\n🚀 Démarrage du pré-entraînement auto-supervisé (Zéro Étiquette)...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Époque [{epoch}/{args.epochs}] (lr={optimizer.param_groups[0]['lr']:.6f}) ---")
        epoch_loss, elapsed = train_one_epoch(
            model, loader, optimizer, device, img_size=args.img_size, mask_ratio=args.mask_ratio, max_batches=args.max_batches
        )
        scheduler.step()
        losses.append(epoch_loss)
        print(f"  👉 Bilan Époque {epoch} ({elapsed:.1f}s) | Loss MSE: {epoch_loss:.4f}")

    # Sauvegarde des poids
    ckpt_path = OUTPUTS_DIR / f"{args.model}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n💾 Poids MAE sauvegardés : {ckpt_path}")

    # Sauvegarde du graphique
    plot_path = OUTPUTS_DIR / "train_mae_convergence.png"
    plot_loss_curve(losses, plot_path)

    print("\n✅ Pré-entraînement MAE achevé avec succès !")


if __name__ == "__main__":
    main()
