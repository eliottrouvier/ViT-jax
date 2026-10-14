"""
ViT-JAX — Phase 3 : MLP Block (Feed-Forward Network) en Flax Linen
Composé de deux couches denses avec projection 4x et activation GELU exacte.
"""

from typing import Optional
import jax.numpy as jnp
from flax import linen as nn


class MLPFlax(nn.Module):
    """
    Bloc MLP à 2 couches linéaires avec expansion intermédiaire.
    Équivalent strict de pytorch/models/mlp.py.
    """
    in_features: int = 768
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None

    def setup(self):
        hidden_dim = self.hidden_features or (self.in_features * 4)
        out_dim = self.out_features or self.in_features
        self.fc1 = nn.Dense(features=hidden_dim, use_bias=True)
        self.fc2 = nn.Dense(features=out_dim, use_bias=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.fc1(x)
        x = nn.gelu(x, approximate=False)
        x = self.fc2(x)
        return x
