"""
ViT-JAX — Phase 3 : Patch Embedding en Flax Linen
Découpe l'image d'entrée en patchs réguliers et les projette linéairement dans l'espace d'embedding.
"""

from typing import Tuple
import jax.numpy as jnp
from flax import linen as nn


class PatchEmbeddingFlax(nn.Module):
    """
    Projection en patchs via Convolution 2D Flax Linen.
    Prend en entrée une image (B, H, W, C) ou (B, C, H, W) et produit (B, num_patches, embed_dim).
    """
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768

    def setup(self):
        self.proj = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Si le tenseur est au format PyTorch NCHW (B, C, H, W), on permute en NHWC
        if x.ndim == 4 and x.shape[1] == self.in_channels and x.shape[-1] != self.in_channels:
            x = jnp.transpose(x, (0, 2, 3, 1))

        x = self.proj(x)  # (B, H_grid, W_grid, embed_dim)
        B, H_g, W_g, D = x.shape
        x = jnp.reshape(x, (B, H_g * W_g, D))  # (B, num_patches, embed_dim)
        return x
