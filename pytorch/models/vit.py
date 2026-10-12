"""
ViT-JAX — Étape 1.2 & 1.6 : Vision Transformer Complet (ViT-B/16)
Référence : Dosovitskiy et al., ICLR 2021 (papiers/vit.pdf)

Modèle Vision Transformer complet assemblant PatchEmbedding, le token [CLS],
l'encodage de position apprenable (avec interpolation bicubique dynamique),
les 12 blocs TransformerEncoder et la tête de classification finale.
"""

from typing import Tuple, Union, Optional, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embed import PatchEmbedding
from .block import TransformerBlock


class VisionTransformer(nn.Module):
    """
    Architecture complète Vision Transformer (ViT).

    Flux de données :
        Image (B, 3, 224, 224)
          ──> PatchEmbedding        : (B, 196, 768)
          ──> Concat [CLS] Token    : (B, 197, 768)
          ──> Ajout Positional Embed: (B, 197, 768)
          ──> 12x TransformerBlock  : (B, 197, 768)
          ──> LayerNorm             : (B, 197, 768)
          ──> Extraction [CLS]      : (B, 768)
          ──> Linear Classifier     : (B, num_classes)
    """
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        # 1.1 Découpage et projection des patchs
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # 1.2 Le token spécial [CLS] apprenable (1, 1, D)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 1.2 Le Positional Embedding apprenable (1, N + 1, D)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 1.6 Empilement des L blocs d'encodeur Transformer (depth = 12 pour ViT-Base)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])

        # Normalisation finale avant la tête de classification
        self.norm = norm_layer(embed_dim)

        # Tête de classification linéaire
        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        # Initialisation des poids selon la recette de Google (troncature normale)
        self._init_weights()

    def _init_weights(self):
        """Initialise pos_embed, cls_token et les couches linéaires/LayerNorm."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_submodule_weights)

    def _init_submodule_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """
        Interpole bicubiquement pos_embed si l'image passée a une résolution différente.
        C'est la méthode que nous avons démontrée théoriquement dans la Fiche 01 !
        """
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == self.patch_embed.img_size[1] and h == self.patch_embed.img_size[0]:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]

        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[1]
        h0 = h // self.patch_embed.patch_size[0]

        # Résolution d'origine en grille carrée
        orig_grid = int(math.sqrt(N))
        patch_pos_embed = patch_pos_embed.reshape(1, orig_grid, orig_grid, dim).permute(0, 3, 1, 2)

        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(h0, w0),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrait la séquence de tokens jusqu'à la couche de normalisation finale."""
        B, C, H, W = x.shape

        # 1. Découpage en patchs : (B, 196, 768)
        x = self.patch_embed(x)

        # 2. Concaténation du [CLS] token devant chaque exemple du batch : (B, 197, 768)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. Ajout de l'encodage de position (avec interpolation si H, W changent)
        pos_embed = self.interpolate_pos_encoding(x, W, H)
        x = x + pos_embed
        x = self.pos_drop(x)

        # 4. Passage à travers les 12 blocs Transformer
        for blk in self.blocks:
            x = blk(x)

        # 5. LayerNorm finale sur tous les tokens
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Extrait le token [CLS] (indice 0) et applique le classifieur."""
        # Seul le token [CLS] est extrait pour la décision de classification globale
        cls_token_out = x[:, 0]
        if pre_logits:
            return cls_token_out
        return self.head(cls_token_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant complète : Image -> Logits."""
        feats = self.forward_features(x)
        logits = self.forward_head(feats)
        return logits

    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne la carte d'attention du dernier bloc Transformer (utile pour DINO)."""
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # Retourne les poids d'attention du dernier bloc
                _, attn = blk(x, return_attention=True)
                return attn


# =============================================================================
# Fonctions de Création (Factory Functions)
# =============================================================================

def vit_base_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """Modèle canonique ViT-B/16 (86 millions de paramètres)."""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_classes=num_classes,
        **kwargs,
    )


def vit_tiny_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """Variante légère ViT-Tiny (5.7 millions de paramètres) pour itération ultra-rapide."""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_classes=num_classes,
        **kwargs,
    )


# =============================================================================
# Test Unitaire Intégré
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Test Unitaire : VisionTransformerB16 Complet")
    print("=" * 60)

    B = 2
    C = 3
    H, W = 224, 224
    num_classes = 100  # ex: CIFAR-100

    model = vit_base_patch16_224(num_classes=num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"• Modèle            : ViT-B/16")
    print(f"• Résolution image  : {H}x{W}")
    print(f"• Nombre de classes : {num_classes}")
    print(f"• Profondeur (blocs): {model.depth}")
    print(f"• Dimension latente : {model.embed_dim}")
    print(f"• Paramètres totaux : {total_params / 1e6:.2f} M (attendu ~86M)")

    dummy_img = torch.randn(B, C, H, W)
    print(f"\n1. Entrée image shape : {dummy_img.shape}")

    # Test forward standard
    logits = model(dummy_img)
    print(f"2. Sortie logits shape : {logits.shape}")
    assert logits.shape == (B, num_classes), f"❌ Forme incorrecte : {logits.shape}"

    # Test backward pass
    loss = logits.sum()
    loss.backward()
    print("3. Rétropropagation    : ✅ Gradients calculés sans erreur.")

    # Test d'extraction d'attention pour DINO
    with torch.no_grad():
        attn = model.get_last_selfattention(dummy_img)
    print(f"4. Matrice attention   : {attn.shape} (attendu: {B}, 12, 197, 197)")
    assert attn.shape == (B, 12, 197, 197)

    print("\n✅ SUCCÈS TOTAL ! ViT-B/16 complet est validé.")
