"""
Tests unitaires de Parité Numérique PyTorch <-> JAX/Flax (Phase 3)
Vérifie composant par composant et sur le modèle complet que les sorties
numériques sont strictement identiques à 1e-5 près avec les poids convertis.
"""

import pytest
import numpy as np
import torch
import jax
import jax.numpy as jnp

from pytorch.models.patch_embed import PatchEmbedding
from pytorch.models.attention import MultiHeadSelfAttention
from pytorch.models.mlp import MlpBlock
from pytorch.models.block import TransformerBlock
from pytorch.models.vit import VisionTransformer, vit_tiny_patch16_224

from jax_flax.models.patch_embed import PatchEmbeddingFlax
from jax_flax.models.attention import MultiHeadSelfAttentionFlax
from jax_flax.models.mlp import MLPFlax
from jax_flax.models.block import TransformerBlockFlax
from jax_flax.models.vit import VisionTransformerFlax, vit_tiny_patch16_224_flax
from jax_flax.weights_converter import torch_to_flax_state_dict


def test_patch_embedding_flax_parity():
    """Vérifie la parité numérique de la projection en patchs (Conv2d)."""
    torch.manual_seed(42)
    B, C, H, W = 2, 3, 32, 32
    patch_size = 16
    embed_dim = 64

    pt_pe = PatchEmbedding(img_size=H, patch_size=patch_size, in_channels=C, embed_dim=embed_dim)
    pt_pe.eval()

    # Création du module Flax et injection des poids PyTorch transposés
    flax_pe = PatchEmbeddingFlax(img_size=H, patch_size=patch_size, in_channels=C, embed_dim=embed_dim)
    w_pt = pt_pe.proj.weight.detach().cpu().numpy()  # (embed_dim, C, kH, kW)
    b_pt = pt_pe.proj.bias.detach().cpu().numpy()
    w_flax = np.transpose(w_pt, (2, 3, 1, 0))        # (kH, kW, C, embed_dim)

    flax_params = {"params": {"proj": {"kernel": jnp.array(w_flax), "bias": jnp.array(b_pt)}}}

    # Entrée identique
    x_pt = torch.randn(B, C, H, W)
    x_jax = jnp.array(x_pt.numpy())

    with torch.no_grad():
        out_pt = pt_pe(x_pt).numpy()
    out_flax = np.array(flax_pe.apply(flax_params, x_jax))

    assert out_pt.shape == out_flax.shape
    diff = np.max(np.abs(out_pt - out_flax))
    assert diff < 1e-5, f"Écart numérique PatchEmbedding trop élevé : {diff}"


def test_attention_flax_parity():
    """Vérifie la parité numérique du mécanisme Multi-Head Self-Attention."""
    torch.manual_seed(42)
    B, N, D, heads = 2, 8, 64, 4

    pt_attn = MultiHeadSelfAttention(dim=D, num_heads=heads, qkv_bias=True)
    pt_attn.eval()

    flax_attn = MultiHeadSelfAttentionFlax(embed_dim=D, num_heads=heads, qkv_bias=True)

    # Conversion des poids
    qkv_w = np.transpose(pt_attn.qkv.weight.detach().cpu().numpy(), (1, 0))
    qkv_b = pt_attn.qkv.bias.detach().cpu().numpy()
    proj_w = np.transpose(pt_attn.proj.weight.detach().cpu().numpy(), (1, 0))
    proj_b = pt_attn.proj.bias.detach().cpu().numpy()

    flax_params = {
        "params": {
            "qkv": {"kernel": jnp.array(qkv_w), "bias": jnp.array(qkv_b)},
            "proj": {"kernel": jnp.array(proj_w), "bias": jnp.array(proj_b)},
        }
    }

    x_pt = torch.randn(B, N, D)
    x_jax = jnp.array(x_pt.numpy())

    with torch.no_grad():
        out_pt = pt_attn(x_pt).numpy()
    out_flax = np.array(flax_attn.apply(flax_params, x_jax))

    assert out_pt.shape == out_flax.shape
    diff = np.max(np.abs(out_pt - out_flax))
    assert diff < 1e-5, f"Écart numérique Self-Attention trop élevé : {diff}"


def test_mlp_flax_parity():
    """Vérifie la parité numérique du bloc MLP avec activation GELU exacte."""
    torch.manual_seed(42)
    B, N, D = 2, 8, 64

    pt_mlp = MlpBlock(in_features=D, hidden_features=D * 4)
    pt_mlp.eval()

    flax_mlp = MLPFlax(in_features=D, hidden_features=D * 4)

    fc1_w = np.transpose(pt_mlp.fc1.weight.detach().cpu().numpy(), (1, 0))
    fc1_b = pt_mlp.fc1.bias.detach().cpu().numpy()
    fc2_w = np.transpose(pt_mlp.fc2.weight.detach().cpu().numpy(), (1, 0))
    fc2_b = pt_mlp.fc2.bias.detach().cpu().numpy()

    flax_params = {
        "params": {
            "fc1": {"kernel": jnp.array(fc1_w), "bias": jnp.array(fc1_b)},
            "fc2": {"kernel": jnp.array(fc2_w), "bias": jnp.array(fc2_b)},
        }
    }

    x_pt = torch.randn(B, N, D)
    x_jax = jnp.array(x_pt.numpy())

    with torch.no_grad():
        out_pt = pt_mlp(x_pt).numpy()
    out_flax = np.array(flax_mlp.apply(flax_params, x_jax))

    diff = np.max(np.abs(out_pt - out_flax))
    assert diff < 1e-5, f"Écart numérique MLP trop élevé : {diff}"


def test_vit_tiny_full_model_parity():
    """Vérifie la parité numérique de bout en bout sur ViT-Tiny complet avec le convertisseur."""
    torch.manual_seed(42)
    B = 2
    C, H, W = 3, 32, 32
    num_classes = 10

    pt_model = VisionTransformer(
        img_size=H,
        patch_size=16,
        in_channels=C,
        embed_dim=64,
        depth=2,
        num_heads=2,
        mlp_ratio=4.0,
        num_classes=num_classes,
    )
    pt_model.eval()

    # Conversion automatique de tous les poids via torch_to_flax_state_dict
    flax_params = torch_to_flax_state_dict(pt_model, num_layers=2)

    flax_model = VisionTransformerFlax(
        img_size=H,
        patch_size=16,
        in_channels=C,
        embed_dim=64,
        depth=2,
        num_heads=2,
        mlp_ratio=4.0,
        num_classes=num_classes,
    )

    x_pt = torch.randn(B, C, H, W)
    x_jax = jnp.array(x_pt.numpy())

    with torch.no_grad():
        out_pt = pt_model(x_pt).numpy()
    out_flax = np.array(flax_model.apply(flax_params, x_jax))

    assert out_pt.shape == out_flax.shape
    diff = np.max(np.abs(out_pt - out_flax))
    print(f"Écart maximal modèle complet ViT (Torch vs Flax) : {diff:.2e}")
    assert diff < 1e-4, f"Écart numérique ViT complet trop élevé : {diff}"


def test_vit_flax_jit_compilation():
    """Vérifie que la compilation @jax.jit fonctionne et produit les mêmes résultats."""
    B, C, H, W = 2, 3, 32, 32
    flax_model = VisionTransformerFlax(
        img_size=H,
        patch_size=16,
        in_channels=C,
        embed_dim=64,
        depth=2,
        num_heads=2,
        num_classes=10,
    )
    rng = jax.random.PRNGKey(0)
    x_dummy = jnp.zeros((B, C, H, W))
    params = flax_model.init(rng, x_dummy)

    # Fonction jittée
    @jax.jit
    def jitted_forward(p, x):
        return flax_model.apply(p, x)

    x_test = jax.random.normal(rng, (B, C, H, W))
    out_normal = flax_model.apply(params, x_test)
    out_jitted = jitted_forward(params, x_test)

    assert jnp.allclose(out_normal, out_jitted, atol=1e-6)
