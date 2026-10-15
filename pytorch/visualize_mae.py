"""
ViT-JAX — Étape 2.7 : Visualisation des Reconstructions MAE (Masked Autoencoder)
Référence : Kaiming He et al., CVPR 2022 (papiers/mae.pdf)

Génère une planche comparative haute résolution :
[ Image Originale ]  |  [ Image Masquée (75%) ]  |  [ Image Reconstruite par MAE ]
et sauvegarde le résultat dans outputs/mae_reconstruction.png.
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn.functional as F

from pytorch.self_supervised import (
    MaskedAutoencoderViT,
    patchify,
    unpatchify,
)

OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_sample_images(num_samples: int = 4) -> torch.Tensor:
    """Charge quelques images d'exemple depuis le dataset local (STL-10 en priorité)."""
    stl10_file = DATA_DIR / "stl10_packed.pt"
    cifar_file = DATA_DIR / "cifar10_packed.pt"

    if stl10_file.exists():
        data = torch.load(stl10_file, weights_only=True)
        # Sélection d'images représentatives du jeu de test STL-10 (animaux, avions, voitures)
        indices = [15, 42, 88, 105][:num_samples]
        imgs = data["test_imgs"][indices].float() / 255.0
        return imgs
    elif cifar_file.exists():
        data = torch.load(cifar_file, weights_only=True)
        imgs = data["test_imgs"][:num_samples].float() / 255.0
        return imgs

    # Fallback
    imgs = torch.zeros(num_samples, 3, 96, 96)
    return imgs


@torch.no_grad()
def run_mae_inference(model, imgs, mask_ratio=0.75, device="cpu", img_size=96):
    """Exécute l'inférence MAE et reconstruit les images."""
    model.eval()
    p = model.patch_size

    # Redimensionnement vers la taille d'entrée du modèle
    if imgs.shape[-1] != img_size:
        imgs = F.interpolate(imgs, size=(img_size, img_size), mode="bicubic", align_corners=False)

    imgs = imgs.to(device)

    # 1. Forward MAE
    loss, pred, mask = model(imgs, mask_ratio=mask_ratio)

    # 2. Patchification de l'image originale
    patches = patchify(imgs, patch_size=p)

    # 3. Création de l'image masquée (patchs masqués peints en gris foncé)
    patches_masked = patches.clone()
    patches_masked[mask == 1] = 0.18  # gris foncé pour les 75% masqués
    imgs_masked = unpatchify(patches_masked, patch_size=p, img_size=img_size)

    # 4. Dénormalisation des prédictions si norm_pix_loss est activé
    if model.norm_pix_loss:
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True)
        pred_unnorm = pred * (var + 1.0e-6) ** 0.5 + mean
    else:
        pred_unnorm = pred

    # 5. Reconstitution complète : patchs visibles originaux + patchs masqués reconstruits
    patches_recon = patches.clone()
    patches_recon[mask == 1] = pred_unnorm[mask == 1]
    imgs_recon = unpatchify(patches_recon, patch_size=p, img_size=img_size)

    # Découpage [0.0, 1.0] pour l'affichage
    imgs = torch.clamp(imgs, 0.0, 1.0).cpu()
    imgs_masked = torch.clamp(imgs_masked, 0.0, 1.0).cpu()
    imgs_recon = torch.clamp(imgs_recon, 0.0, 1.0).cpu()

    return imgs, imgs_masked, imgs_recon, loss.item()


def plot_triptych(imgs, imgs_masked, imgs_recon, out_path, num_samples=4):
    """Génère la planche triptyque : Originale | Masquée (75%) | Reconstituée."""
    fig, axes = plt.subplots(num_samples, 3, figsize=(9.5, 3.2 * num_samples))

    titles = ["1. Image Originale (STL-10)", "2. Masquée (75% Caché)", "3. Reconstruction MAE"]

    for row in range(num_samples):
        # Conversion PyTorch (C, H, W) -> Matplotlib (H, W, C)
        orig = imgs[row].permute(1, 2, 0).numpy()
        masked = imgs_masked[row].permute(1, 2, 0).numpy()
        recon = imgs_recon[row].permute(1, 2, 0).numpy()

        for col, (im, title) in enumerate(zip([orig, masked, recon], titles)):
            axes[row, col].imshow(im)
            if row == 0:
                axes[row, col].set_title(title, fontsize=12, fontweight="bold", pad=10)
            axes[row, col].axis("off")

    plt.suptitle("Masked Autoencoders (MAE) — Reconstruction Auto-Supervisée", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n🎨 Planche triptyque MAE sauvegardée : {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualisation des reconstructions MAE")
    parser.add_argument("--checkpoint", type=str, default="outputs/mae_vit_tiny.pt",
                        help="Chemin vers le fichier de poids (ex: outputs/mae_vit_tiny.pt)")
    parser.add_argument("--num-samples", type=int, default=4, help="Nombre d'images à afficher")
    parser.add_argument("--img-size", type=int, default=96, help="Résolution d'image (96 pour STL-10)")
    parser.add_argument("--patch-size", type=int, default=8, help="Taille des patchs (8 pour 144 tokens)")
    parser.add_argument("--mask-ratio", type=float, default=0.75, help="Ratio de masquage (0.75)")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Instanciation du modèle MAE
    model = MaskedAutoencoderViT(
        img_size=args.img_size, patch_size=args.patch_size, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4, norm_pix_loss=True
    )

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"Chargement des poids depuis {ckpt_path}...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        print(f"⚠️ Checkpoint {ckpt_path} non trouvé, utilisation des poids initiaux.")

    model = model.to(device)

    # Chargement d'exemples
    imgs = load_sample_images(num_samples=args.num_samples)

    # Inférence MAE
    imgs_orig, imgs_masked, imgs_recon, loss = run_mae_inference(
        model, imgs, mask_ratio=args.mask_ratio, device=device, img_size=args.img_size
    )

    print(f"Perte de reconstruction MSE sur les 75% masqués : {loss:.4f}")

    # Sauvegarde du graphique
    out_path = OUTPUTS_DIR / "mae_reconstruction.png"
    plot_triptych(imgs_orig, imgs_masked, imgs_recon, out_path, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
