"""
ViT-JAX — Étape 1.5 : Bloc Transformer Encoder (TransformerBlock)
Référence : Dosovitskiy et al. (papiers/vit.pdf)

Bloc d'encodage complet avec Pre-LayerNorm, Multi-Head Self-Attention,
connexions résiduelles (+ x) et MLP à deux couches.
"""

from typing import Tuple, Optional, Callable
import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .mlp import MlpBlock


class TransformerBlock(nn.Module):
    """
    Bloc encodeur du Vision Transformer avec architecture Pre-LayerNorm.

    Schéma :
        x ────────┬──────────────────────────(+)───┬──────────────────────────(+)──> Sortie
                  │                           ▲    │                           ▲
                  ▼                           │    ▼                           │
              [Norm 1] ──> [Attention MHSA] ──┘ [Norm 2] ──> [Bloc MLP (GELU)] ──┘

    Arguments :
        dim: Dimension des tokens (D = 768).
        num_heads: Nombre de têtes d'attention (h = 12).
        mlp_ratio: Ratio d'expansion pour le MLP (4.0 -> 3072).
        qkv_bias: Si True, active le biais sur les projections QKV.
        drop: Taux de dropout général (MLP et projections).
        attn_drop: Taux de dropout sur les matrices d'attention.
        act_layer: Fonction d'activation (nn.GELU).
        norm_layer: Type de couche de normalisation (nn.LayerNorm).
    """
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        # Pre-LayerNorm pour la branche d'attention
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Pre-LayerNorm pour la branche MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpBlock(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Passe avant (Forward Pass).

        Arguments :
            x: Tenseur de tokens de forme (B, N, D)
            return_attention: Si True, retourne la carte d'attention softmax de ce bloc

        Retourne :
            out: Tenseur mis à jour de forme (B, N, D)
            (optionnel) attn: Matrice d'attention (B, num_heads, N, N)
        """
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        else:
            # Pre-LN : la normalisation est appliquée AVANT l'attention et le MLP
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


# =============================================================================
# Test Unitaire Intégré
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Test Unitaire : TransformerBlock (Pre-LayerNorm)")
    print("=" * 60)

    B = 2
    N = 197
    D = 768
    heads = 12

    block = TransformerBlock(dim=D, num_heads=heads, mlp_ratio=4.0)

    dummy_tokens = torch.randn(B, N, D)
    print(f"1. Entrée tokens shape : {dummy_tokens.shape}")

    out = block(dummy_tokens)
    print(f"2. Sortie tokens shape : {out.shape}")
    assert out.shape == (B, N, D), f"❌ Forme incorrecte : {out.shape}"

    out_with_attn, attn = block(dummy_tokens, return_attention=True)
    print(f"3. Matrice d'attention : {attn.shape}")
    assert attn.shape == (B, heads, N, N), f"❌ Forme attention incorrecte"

    print("\n✅ SUCCÈS ! TransformerBlock est validé.")
