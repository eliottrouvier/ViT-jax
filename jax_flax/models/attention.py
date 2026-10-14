"""
ViT-JAX — Phase 3 : Multi-Head Self-Attention en Flax Linen
Calcul d'attention multi-têtes vectorisé avec projections Q, K, V et mise à l'échelle.
"""

import math
from typing import Tuple, Union
import jax
import jax.numpy as jnp
from flax import linen as nn


class MultiHeadSelfAttentionFlax(nn.Module):
    """
    Mécanisme d'attention multi-têtes conforme à Vaswani et al. & Dosovitskiy et al.
    Strictement équivalent à la version PyTorch pour une parité à 1e-5.
    """
    embed_dim: int = 768
    num_heads: int = 12
    qkv_bias: bool = True

    def setup(self):
        self.qkv = nn.Dense(features=self.embed_dim * 3, use_bias=self.qkv_bias)
        self.proj = nn.Dense(features=self.embed_dim, use_bias=True)

    def __call__(
        self, x: jnp.ndarray, return_attention: bool = False
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        B, N, C = x.shape
        head_dim = self.embed_dim // self.num_heads
        scale = 1.0 / math.sqrt(head_dim)

        # 1. Projection unifiée Q, K, V : (B, N, 3 * embed_dim)
        qkv = self.qkv(x)

        # 2. Reshape et séparation en (3, B, num_heads, N, head_dim)
        qkv = jnp.reshape(qkv, (B, N, 3, self.num_heads, head_dim))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 3. Scaled Dot-Product Attention : Softmax((Q @ K^T) / sqrt(d))
        attn_scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)

        # 4. Pondération des valeurs V
        out = jnp.matmul(attn_weights, v)  # (B, num_heads, N, head_dim)
        out = jnp.swapaxes(out, 1, 2)     # (B, N, num_heads, head_dim)
        out = jnp.reshape(out, (B, N, C))  # (B, N, embed_dim)

        # 5. Projection linéaire finale
        out = self.proj(out)

        if return_attention:
            return out, attn_weights
        return out
