"""
Tests unitaires pour Masked Autoencoders (MAE) - Phase 2A
Vérifie le découpage en patchs, l'algorithme de masquage 75%,
l'asymétrie encodeur/décodeur, la rétropropagation et la parité de reconstruction.
"""

import pytest
import torch

from pytorch.self_supervised import (
    MaskedAutoencoderViT,
    mae_vit_tiny_patch16_224,
    patchify,
    unpatchify,
)


def test_patchify_and_unpatchify():
    """Vérifie que patchify et unpatchify sont des bijections exactes."""
    B, C, H, W = 2, 3, 224, 224
    p = 16
    imgs = torch.randn(B, C, H, W)

    # 1. Image vers patchs aplatis
    patches = patchify(imgs, patch_size=p)
    num_patches = (H // p) * (W // p)
    patch_dim = p * p * C
    assert patches.shape == (B, num_patches, patch_dim), f"Forme inattendue : {patches.shape}"

    # 2. Patchs aplatis vers image
    imgs_recon = unpatchify(patches, patch_size=p, img_size=H)
    assert imgs_recon.shape == imgs.shape

    # 3. Vérification de la parité numérique exacte
    diff = (imgs - imgs_recon).abs().max().item()
    assert diff < 1e-6, f"Erreur de reconstruction patchify/unpatchify trop élevée : {diff}"


def test_random_masking_75_percent():
    """Vérifie l'algorithme de masquage aléatoire par double argsort."""
    model = mae_vit_tiny_patch16_224()
    B, N, D = 4, 196, 192
    x = torch.randn(B, N, D)
    mask_ratio = 0.75

    x_masked, mask, ids_restore = model.random_masking(x, mask_ratio=mask_ratio)

    expected_keep = int(N * (1 - mask_ratio))  # 49 patchs
    assert x_masked.shape == (B, expected_keep, D), f"Forme x_masked inattendue : {x_masked.shape}"
    assert mask.shape == (B, N), f"Forme mask inattendue : {mask.shape}"
    assert ids_restore.shape == (B, N), f"Forme ids_restore inattendue : {ids_restore.shape}"

    # Vérification du nombre exact de 0 (visibles) et 1 (masqués)
    for b in range(B):
        num_zeros = (mask[b] == 0).sum().item()
        num_ones = (mask[b] == 1).sum().item()
        assert num_zeros == expected_keep, f"Attendu {expected_keep} patchs visibles, obtenu {num_zeros}"
        assert num_ones == N - expected_keep, f"Attendu {N - expected_keep} patchs masqués, obtenu {num_ones}"


def test_mae_encoder_forward():
    """Vérifie que l'encodeur ne traite que les 25% de patchs visibles (+ 1 CLS)."""
    model = mae_vit_tiny_patch16_224()
    imgs = torch.randn(2, 3, 224, 224)

    latent, mask, ids_restore = model.forward_encoder(imgs, mask_ratio=0.75)

    # 49 patchs visibles + 1 token [CLS] = 50 tokens
    assert latent.shape == (2, 50, 192), f"Forme encodeur inattendue : {latent.shape}"
    assert mask.shape == (2, 196)
    assert ids_restore.shape == (2, 196)


def test_mae_decoder_forward():
    """Vérifie que le décodeur reconstruit les 196 patchs de pixels après unshuffle."""
    model = mae_vit_tiny_patch16_224()
    imgs = torch.randn(2, 3, 224, 224)

    latent, mask, ids_restore = model.forward_encoder(imgs, mask_ratio=0.75)
    pred = model.forward_decoder(latent, ids_restore)

    # 196 patchs de 16x16x3 = 768 pixels chacun
    assert pred.shape == (2, 196, 768), f"Forme prédiction décodeur inattendue : {pred.shape}"


def test_mae_full_forward_and_loss_backward():
    """Vérifie la passe complète et la propagation des gradients sur l'encodeur et le décodeur."""
    model = mae_vit_tiny_patch16_224()
    imgs = torch.randn(2, 3, 224, 224, requires_grad=False)

    loss, pred, mask = model(imgs, mask_ratio=0.75)

    assert isinstance(loss.item(), float)
    assert loss.item() > 0.0
    assert pred.shape == (2, 196, 768)
    assert mask.shape == (2, 196)

    # Rétropropagation des gradients
    loss.backward()

    # Vérification que l'encodeur et le décodeur ont tous reçu des gradients valides
    assert model.patch_embed.proj.weight.grad is not None
    assert model.blocks[0].attn.qkv.weight.grad is not None
    assert model.decoder_blocks[0].attn.qkv.weight.grad is not None
    assert model.mask_token.grad is not None
    assert model.decoder_pred.weight.grad is not None
