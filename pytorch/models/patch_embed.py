"""
ViT-JAX — Étape 1.1 : Patch Embedding
Référence : Dosovitskiy et al., ICLR 2021 (papiers/vit.pdf)

Ce module découpe une image 2D (B, C, H, W) en une grille de patchs non chevauchants
et projette linéairement chaque patch dans un espace latent de dimension D (embed_dim).
"""

from typing import Tuple, Union, Optional, Callable
import torch
import torch.nn as nn


def _to_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convertit un entier n en tuple (n, n) si nécessaire."""
    return x if isinstance(x, tuple) else (x, x)


class PatchEmbedding(nn.Module):
    """
    Découpage d'image 2D en patchs et projection linéaire vers un espace latent.

    Arguments :
        img_size: Taille de l'image d'entrée (int ou tuple H, W), défaut: 224.
        patch_size: Taille d'un patch (int ou tuple P, P), défaut: 16.
        in_channels: Nombre de canaux de l'image (RGB = 3), défaut: 3.
        embed_dim: Dimension de projection du Transformer (D), défaut: 768.
        norm_layer: Couche de normalisation optionnelle appliquée aux tokens projetés.
    """
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ):
        super().__init__()
        self.img_size = _to_2tuple(img_size)
        self.patch_size = _to_2tuple(patch_size)
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Vérification arithmétique de divisibilité
        assert self.img_size[0] % self.patch_size[0] == 0, (
            f"La hauteur ({self.img_size[0]}) doit être divisible par le patch ({self.patch_size[0]})"
        )
        assert self.img_size[1] % self.patch_size[1] == 0, (
            f"La largeur ({self.img_size[1]}) doit être divisible par le patch ({self.patch_size[1]})"
        )

        # Calcul de la grille spatiale et du nombre total de patchs N
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Projection convolutive non-chevauchante
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        # Normalisation optionnelle post-projection
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant (Forward Pass).

        Entrée :
            x : Tenseur image de forme (B, in_channels, H, W)
        Sortie :
            Tenseur de tokens de forme (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape

        # Vérifications de cohérence des dimensions
        assert C == self.in_channels, (
            f"Nombre de canaux incorrect : reçu {C}, attendu {self.in_channels}"
        )
        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, (
            f"Image ({H}, {W}) non divisible par la taille du patch ({self.patch_size})"
        )

        # 1. Convolution 2D : (B, 3, 224, 224) -> (B, 768, 14, 14)
        x = self.proj(x)

        # 2. Aplatissement spatial et transposition :
        # x.flatten(2)      : (B, 768, 14, 14) -> (B, 768, 196)
        # .transpose(1, 2)  : (B, 768, 196)    -> (B, 196, 768)
        x = x.flatten(2).transpose(1, 2)

        # 3. Normalisation optionnelle le long de la dimension embed_dim
        x = self.norm(x)

        return x


# =============================================================================
# Test Unitaire Intégré
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Test Unitaire : PatchEmbedding (ViT-B/16)")
    print("=" * 60)

    B, C, H, W = 2, 3, 224, 224
    P = 16
    D = 768

    patch_embed = PatchEmbedding(
        img_size=H,
        patch_size=P,
        in_channels=C,
        embed_dim=D,
        norm_layer=nn.LayerNorm,
    )

    print(f"• Résolution image    : {H}x{W}")
    print(f"• Taille de patch     : {P}x{P}")
    print(f"• Grille de patchs    : {patch_embed.grid_size[0]}x{patch_embed.grid_size[1]}")
    print(f"• Nombre total patchs : {patch_embed.num_patches}")
    print(f"• Dimension embed (D) : {patch_embed.embed_dim}")

    # Tenseur d'entrée aléatoire
    dummy_input = torch.randn(B, C, H, W)
    print(f"\n1. Entrée dummy shape : {dummy_input.shape}")

    try:
        output = patch_embed(dummy_input)
        print(f"2. Sortie shape       : {output.shape}")

        expected_shape = (B, 196, D)
        assert output.shape == expected_shape, (
            f"❌ Forme incorrecte ! Reçu {output.shape}, attendu {expected_shape}"
        )
        print("\n✅ SUCCÈS ! La forme (B, 196, 768) est rigoureusement validée.")
    except Exception as e:
        print(f"\n❌ Erreur rencontrée : {e}")
