"""
Suite de tests unitaires formelle pour ViT-B/16 (Étape 1.7)
Exécutable directement avec pytest ou python -m tests.test_vit
"""

import pytest
import torch
import torch.nn as nn

from pytorch.models import (
    PatchEmbedding,
    MultiHeadSelfAttention,
    MlpBlock,
    TransformerBlock,
    VisionTransformer,
    vit_base_patch16_224,
    vit_tiny_patch16_224,
)


def test_patch_embedding():
    B, C, H, W = 2, 3, 224, 224
    patch_embed = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=768)
    x = torch.randn(B, C, H, W)
    out = patch_embed(x)
    assert out.shape == (B, 196, 768), f"Erreur PatchEmbed shape: {out.shape}"


def test_attention_mechanism():
    B, N, D = 2, 197, 768
    attn = MultiHeadSelfAttention(dim=D, num_heads=12)
    x = torch.randn(B, N, D)
    out, weights = attn(x, return_attention=True)
    assert out.shape == (B, N, D)
    assert weights.shape == (B, 12, N, N)
    # Vérification propriété fondamentale de probabilité Softmax (somme = 1)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_mlp_block():
    B, N, D = 2, 197, 768
    mlp = MlpBlock(in_features=D, hidden_features=D * 4)
    x = torch.randn(B, N, D)
    out = mlp(x)
    assert out.shape == (B, N, D)


def test_transformer_block():
    B, N, D = 2, 197, 768
    block = TransformerBlock(dim=D, num_heads=12)
    x = torch.randn(B, N, D)
    out = block(x)
    assert out.shape == (B, N, D)


def test_vit_b16_full_architecture():
    B, C, H, W = 2, 3, 224, 224
    num_classes = 100
    model = vit_base_patch16_224(num_classes=num_classes)

    # 1. Vérification du décompte de paramètres (~85.8M pour ViT-B/16 avec 100 classes)
    total_params = sum(p.numel() for p in model.parameters())
    assert 85_000_000 < total_params < 87_000_000, f"Nombre de paramètres inattendu: {total_params}"

    # 2. Forward pass
    x = torch.randn(B, C, H, W)
    logits = model(x)
    assert logits.shape == (B, num_classes)

    # 3. Backward pass & gradients
    loss = logits.sum()
    loss.backward()
    assert model.patch_embed.proj.weight.grad is not None
    assert model.cls_token.grad is not None
    assert model.pos_embed.grad is not None
    assert not torch.isnan(model.patch_embed.proj.weight.grad).any()


def test_vit_dynamic_resolution_interpolation():
    """Vérifie que l'interpolation bicubique fonctionne sur une résolution non standard."""
    model = vit_tiny_patch16_224(num_classes=10)
    # Image plus petite : 112x112 (7x7 = 49 patchs au lieu de 196)
    small_img = torch.randn(1, 3, 112, 112)
    out = model(small_img)
    assert out.shape == (1, 10), "L'interpolation bicubique dynamique a échoué sur 112x112 !"


if __name__ == "__main__":
    print("Exécution des tests de validation Phase 1...")
    test_patch_embedding()
    print("  ✓ test_patch_embedding passé")
    test_attention_mechanism()
    print("  ✓ test_attention_mechanism passé")
    test_mlp_block()
    print("  ✓ test_mlp_block passé")
    test_transformer_block()
    print("  ✓ test_transformer_block passé")
    test_vit_b16_full_architecture()
    print("  ✓ test_vit_b16_full_architecture passé")
    test_vit_dynamic_resolution_interpolation()
    print("  ✓ test_vit_dynamic_resolution_interpolation passé")
    print("\n🎉 TOUS LES TESTS DE L'ÉTAPE 1.7 SONT VALIDÉS AVEC SUCCÈS !")
