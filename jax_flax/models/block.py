"""
ViT-JAX — Phase 3 : Bloc Encodeur Transformer en Flax Linen
Assemble Pre-LayerNorm, Self-Attention et Bloc MLP avec connexions résiduelles.
"""

from typing import Tuple, Union
import jax.numpy as jnp
from flax import linen as nn

from jax_flax.models.attention import MultiHeadSelfAttentionFlax
from jax_flax.models.mlp import MLPFlax


class TransformerBlockFlax(nn.Module):
    """
    Bloc Encodeur Transformer pré-normé (Pre-LayerNorm).
    Structure :
      x -> Norm1 -> Attn -> (+) -> Norm2 -> MLP -> (+)
    """
    embed_dim: int = 768
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True

    def setup(self):
        self.norm1 = nn.LayerNorm(epsilon=1e-5)
        self.attn = MultiHeadSelfAttentionFlax(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
        )
        self.norm2 = nn.LayerNorm(epsilon=1e-5)
        self.mlp = MLPFlax(
            in_features=self.embed_dim,
            hidden_features=int(self.embed_dim * self.mlp_ratio),
        )

    def __call__(
        self, x: jnp.ndarray, return_attention: bool = False
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        norm1_out = self.norm1(x)
        if return_attention:
            attn_out, attn_weights = self.attn(norm1_out, return_attention=True)
        else:
            attn_out = self.attn(norm1_out, return_attention=False)
        x = x + attn_out

        norm2_out = self.norm2(x)
        mlp_out = self.mlp(norm2_out)
        x = x + mlp_out

        if return_attention:
            return x, attn_weights
        return x
