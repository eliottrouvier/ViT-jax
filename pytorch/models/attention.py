"""
ViT-JAX — Étape 1.3 : Multi-Head Self-Attention (MHSA)
Référence : Vaswani et al. (papiers/attention.pdf) & Dosovitskiy et al. (papiers/vit.pdf)

Implémentation de l'attention multi-têtes avec projection QKV fusionnée,
mise à l'échelle par 1/sqrt(d_k), Softmax et export optionnel des poids d'attention.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Mécanisme d'Attention Multi-Têtes (Multi-Head Self-Attention).

    Mathématiques :
        Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V

    Arguments :
        dim: Dimension totale des représentations (D = 768 pour ViT-B).
        num_heads: Nombre de têtes d'attention parallèles (h = 12 pour ViT-B).
        qkv_bias: Si True, ajoute un biais aux projections linéaires Q, K, V.
        attn_drop: Probabilité de dropout sur la matrice d'attention softmax.
        proj_drop: Probabilité de dropout sur la projection linéaire de sortie.
    """
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"La dimension ({dim}) doit être divisible par le nombre de têtes ({num_heads})"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # d_k = 768 // 12 = 64
        self.scale = 1.0 / math.sqrt(self.head_dim)  # 1 / sqrt(64) = 0.125

        # Projection fusionnée Q, K, V en une seule couche linéaire (D -> 3 * D)
        # Permet un parallélisme GPU/MPS optimal
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Passe avant (Forward Pass).

        Arguments :
            x: Tenseur de tokens de forme (B, N, D)
            return_attention: Si True, retourne aussi la matrice d'attention (B, num_heads, N, N)

        Retourne :
            out: Tenseur de tokens de forme (B, N, D)
            (optionnel) attn: Matrice d'attention de forme (B, num_heads, N, N)
        """
        B, N, D = x.shape
        assert D == self.dim, f"Dimension incorrecte : reçue {D}, attendue {self.dim}"

        # 1. Projection linéaire QKV : (B, N, D) -> (B, N, 3 * D)
        # Puis reshaping vers (B, N, 3, num_heads, head_dim)
        # Puis permutation vers (3, B, num_heads, N, head_dim)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )

        # Décomposition en Q, K, V chacun de forme (B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Scaled Dot-Product Attention :
        # Q @ K^T : (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        # 3. Normalisation Softmax le long du dernier axe (sur les clés N)
        attn = attn_scores.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 4. Pondération des valeurs V :
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        out = attn @ v

        # 5. Concaténation des têtes et projection linéaire finale :
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, D)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attention:
            return out, attn
        return out


# =============================================================================
# Test Unitaire Intégré
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Test Unitaire : MultiHeadSelfAttention (ViT-B/16)")
    print("=" * 60)

    B = 2
    N = 197  # 196 patchs + 1 [CLS] token
    D = 768  # ViT-Base
    heads = 12

    mhsa = MultiHeadSelfAttention(dim=D, num_heads=heads)

    print(f"• Dimension D         : {D}")
    print(f"• Nombre de têtes     : {heads}")
    print(f"• Dimension par tête  : {mhsa.head_dim} (d_k)")
    print(f"• Facteur d'échelle   : {mhsa.scale:.4f} (1 / sqrt(d_k))")

    dummy_tokens = torch.randn(B, N, D)
    print(f"\n1. Entrée tokens shape : {dummy_tokens.shape}")

    # Test avec sortie standard
    out = mhsa(dummy_tokens)
    print(f"2. Sortie tokens shape : {out.shape}")
    assert out.shape == (B, N, D), f"❌ Forme incorrecte : {out.shape}"

    # Test avec export des poids d'attention (pour DINO)
    out_with_attn, attn_matrix = mhsa(dummy_tokens, return_attention=True)
    print(f"3. Matrice attention   : {attn_matrix.shape}")
    assert attn_matrix.shape == (B, heads, N, N), f"❌ Forme attention incorrecte : {attn_matrix.shape}"

    # Vérification que la somme des probabilités d'attention vaut 1 le long du dernier axe
    sums = attn_matrix.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "❌ Softmax ne somme pas à 1 !"

    print("\n✅ SUCCÈS ! MultiHeadSelfAttention est mathématiquement validé.")
