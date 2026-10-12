"""
ViT-JAX — Étape 1.4 : Bloc MLP (Feed-Forward Network)
Référence : Dosovitskiy et al. (papiers/vit.pdf)

Bloc perceptron multicouche à deux couches linéaires avec expansion 4x,
fonction d'activation non-linéaire GELU et dropout.
"""

from typing import Optional, Callable
import torch
import torch.nn as nn


class MlpBlock(nn.Module):
    """
    Bloc Feed-Forward (MLP) avec expansion 4x et activation GELU.

    Structure :
        Linear(D -> 4*D) -> GELU -> Dropout -> Linear(4*D -> D) -> Dropout

    Arguments :
        in_features: Dimension d'entrée (D = 768).
        hidden_features: Dimension de la couche cachée (défaut: 4 * in_features = 3072).
        out_features: Dimension de sortie (défaut: in_features = 768).
        act_layer: Fonction d'activation (défaut: nn.GELU).
        drop: Taux de dropout (défaut: 0.0).
    """
    def __init__(
        self,
        in_features: int = 768,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant (Forward Pass).
        Entrée : (B, N, in_features)
        Sortie : (B, N, out_features)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# =============================================================================
# Test Unitaire Intégré
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Test Unitaire : MlpBlock (ViT-B/16)")
    print("=" * 60)

    B = 2
    N = 197
    D = 768
    hidden_dim = D * 4  # 3072

    mlp = MlpBlock(in_features=D, hidden_features=hidden_dim, drop=0.1)

    print(f"• Dimension in/out    : {D}")
    print(f"• Dimension cachée    : {hidden_dim} (expansion 4x)")
    print(f"• Activation          : GELU")

    dummy_tokens = torch.randn(B, N, D)
    print(f"\n1. Entrée tokens shape : {dummy_tokens.shape}")

    out = mlp(dummy_tokens)
    print(f"2. Sortie tokens shape : {out.shape}")

    assert out.shape == (B, N, D), f"❌ Forme incorrecte : {out.shape}"
    print("\n✅ SUCCÈS ! MlpBlock est validé.")
