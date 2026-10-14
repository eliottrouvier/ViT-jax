"""
ViT-JAX — Phase 3 : Vision Transformer (ViT) Complet en Flax Linen
Référence : Dosovitskiy et al., ICLR 2021 (papiers/vit.pdf)
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

Architecture fonctionnelle pure compilable via @jax.jit avec XLA.
"""

from typing import Tuple, Union, Optional
import jax.numpy as jnp
from flax import linen as nn

from jax_flax.models.patch_embed import PatchEmbeddingFlax
from jax_flax.models.block import TransformerBlockFlax


class VisionTransformerFlax(nn.Module):
    """
    Vision Transformer canonique implémenté en Flax Linen.
    Strictement équivalent à la classe VisionTransformer en PyTorch (pytorch/models/vit.py).
    """
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True

    def setup(self):
        self.num_patches = (self.img_size // self.patch_size) ** 2

        # 1. Patch Embedding
        self.patch_embed = PatchEmbeddingFlax(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
        )

        # 2. Paramètres apprenables : CLS token et Position Embeddings
        self.cls_token = self.param(
            "cls_token",
            nn.initializers.zeros,
            (1, 1, self.embed_dim),
        )
        self.pos_embed = self.param(
            "pos_embed",
            nn.initializers.zeros,
            (1, self.num_patches + 1, self.embed_dim),
        )

        # 3. Blocs Encodeurs Transformer empilés
        self.blocks = [
            TransformerBlockFlax(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                name=f"blocks_{i}",
            )
            for i in range(self.depth)
        ]

        # 4. LayerNorm finale & Tête de classification
        self.norm = nn.LayerNorm(epsilon=1e-5)
        if self.num_classes > 0:
            self.head = nn.Dense(features=self.num_classes, use_bias=True)
        else:
            self.head = None

    def __call__(
        self, x: jnp.ndarray, return_attention: bool = False
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        B = x.shape[0]

        # 1. Projection en patchs
        x = self.patch_embed(x)  # (B, N, embed_dim)

        # 2. Concaténation du [CLS] token
        cls_tokens = jnp.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)  # (B, N+1, embed_dim)

        # 3. Ajout des encodages positionnels
        x = x + self.pos_embed

        # 4. Passage à travers les blocs Transformer
        last_attn = None
        for i, block in enumerate(self.blocks):
            if return_attention and i == len(self.blocks) - 1:
                x, last_attn = block(x, return_attention=True)
            else:
                x = block(x)

        # 5. Normalisation finale et extraction du [CLS] token
        x = self.norm(x)
        cls_out = x[:, 0]  # Représentation globale

        # 6. Tête de classification
        if self.head is not None:
            logits = self.head(cls_out)
        else:
            logits = cls_out

        if return_attention:
            return logits, last_attn
        return logits


# -----------------------------------------------------------------------------
# Fonctions Fabriques (Model Factories)
# -----------------------------------------------------------------------------

def vit_base_patch16_224_flax(num_classes: int = 1000, **kwargs) -> VisionTransformerFlax:
    """Modèle canonique ViT-B/16 en Flax Linen (86M paramètres)."""
    return VisionTransformerFlax(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=num_classes,
        **kwargs,
    )


def vit_tiny_patch16_224_flax(num_classes: int = 1000, **kwargs) -> VisionTransformerFlax:
    """Modèle ultra-léger ViT-Tiny en Flax Linen (5.7M paramètres)."""
    return VisionTransformerFlax(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        num_classes=num_classes,
        **kwargs,
    )
