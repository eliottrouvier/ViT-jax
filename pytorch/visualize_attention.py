"""
ViT-JAX — Phase 2B : Visualisation des Cartes d'Attention Émergentes DINO (Étape 2.14)
Référence : Caron et al., "Emerging Properties in Self-Supervised Vision Transformers"

Extrait les cartes d'attention du [CLS] token de la dernière couche du Vision Transformer.
Sans aucune étiquette ni supervision sémantique, DINO apprend spontanément à segmenter
les objets saillants au sein des têtes d'attention.
Sauvegarde : outputs/dino_attention_maps.png
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn.functional as F

from pytorch.self_supervised import (
    dino_vit_tiny,
    dino_vit_base,
    DINOModel,
)

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dino_model(ckpt_path: Path, img_size: int = 64, device: torch.device = torch.device("cpu")) -> DINOModel:
    """Charge le modèle DINO (poids Teacher ou Student)."""
    student, teacher = dino_vit_tiny(out_dim=1024, img_size=img_size)
    if ckpt_path.exists():
        print(f"Chargement du checkpoint : {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if "teacher" in state:
            teacher.load_state_dict(state["teacher"])
        else:
            teacher.load_state_dict(state)
    else:
        print(f"⚠️ Checkpoint non trouvé à {ckpt_path}, utilisation des poids initialisés aléatoirement.")
    teacher.to(device)
    teacher.eval()
    return teacher


def get_sample_images(num_samples: int = 4, img_size: int = 64) -> torch.Tensor:
    """Récupère des images d'exemples depuis CIFAR-10."""
    packed_file = DATA_DIR / "cifar10_packed.pt"
    if packed_file.exists():
        data = torch.load(packed_file, weights_only=True)
        # Sélection de quelques images représentatives du set de test
        indices = [12, 45, 88, 105][:num_samples]
        imgs = data["test_imgs"][indices].float() / 255.0
    else:
        imgs = torch.rand(num_samples, 3, 32, 32)

    if imgs.shape[-1] != img_size:
        imgs = F.interpolate(imgs, size=(img_size, img_size), mode="bicubic", align_corners=False)
    return imgs


def generate_attention_maps(
    model: DINOModel,
    images: torch.Tensor,
    img_size: int = 64,
    patch_size: int = 16,
    out_path: Path = OUTPUTS_DIR / "dino_attention_maps.png",
):
    """
    Extrait l'attention du [CLS] token et génère les superpositions thermiques.
    """
    B, C, H, W = images.shape
    device = next(model.parameters()).device
    images_gpu = images.to(device)

    # Extraction des matrices d'attention : (B, num_heads, N, N)
    with torch.no_grad():
        attn = model.get_last_selfattention(images_gpu)

    num_heads = attn.shape[1]
    # L'attention du [CLS] token vers tous les patchs spatiaux (indice 0 -> 1:)
    cls_attn = attn[:, :, 0, 1:]  # (B, num_heads, num_patches)

    w_featmap = H // patch_size
    h_featmap = W // patch_size
    cls_attn = cls_attn.reshape(B, num_heads, h_featmap, w_featmap)

    # Interpolation bilinéaire à la résolution originale de l'image
    cls_attn = F.interpolate(cls_attn, size=(H, W), mode="bilinear", align_corners=False)
    cls_attn = cls_attn.cpu().numpy()

    # Normalisation min-max par tête pour le contraste visuel
    for b in range(B):
        for h in range(num_heads):
            m, M = cls_attn[b, h].min(), cls_attn[b, h].max()
            cls_attn[b, h] = (cls_attn[b, h] - m) / (M - m + 1e-8)

    # Création du panneau de visualisation
    # Colonnes : Image originale | Tête 1 | Tête 2 | Tête 3 | Moyenne superposée
    cols = 2 + num_heads
    fig, axes = plt.subplots(B, cols, figsize=(3 * cols, 3 * B))
    if B == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(B):
        img_np = images[i].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0.0, 1.0)

        # 1. Image originale
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Image {i+1}", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # 2. Chaque tête d'attention individuelle
        for h in range(num_heads):
            axes[i, 1 + h].imshow(cls_attn[i, h], cmap="inferno")
            axes[i, 1 + h].set_title(f"Tête {h+1}", fontsize=10)
            axes[i, 1 + h].axis("off")

        # 3. Superposition moyenne (Heatmap + Image originale)
        mean_attn = cls_attn[i].mean(axis=0)
        axes[i, cols - 1].imshow(img_np)
        axes[i, cols - 1].imshow(mean_attn, cmap="viridis", alpha=0.55)
        axes[i, cols - 1].set_title("Overlay Segmentation", fontsize=10, fontweight="bold", color="#8b5cf6")
        axes[i, cols - 1].axis("off")

    plt.suptitle("Propriétés Émergentes DINO : Cartes d'Attention [CLS] Auto-Supervisées", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🎨 Cartes d'attention DINO sauvegardées : {out_path}")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt_path = OUTPUTS_DIR / "dino_vit_tiny.pt"
    img_size = 64

    model = load_dino_model(ckpt_path, img_size=img_size, device=device)
    images = get_sample_images(num_samples=4, img_size=img_size)
    generate_attention_maps(model, images, img_size=img_size, patch_size=16)


if __name__ == "__main__":
    main()
