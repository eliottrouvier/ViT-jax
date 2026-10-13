"""
ViT-JAX — Phase 2A : Masked Autoencoder (MAE)
Référence : Kaiming He et al., CVPR 2022 (papiers/mae.pdf)
"Masked Autoencoders Are Scalable Vision Learners"

Implémentation de l'architecture MAE asymétrique :
- Encodeur lourd (ViT-B/16 ou ViT-Tiny) ne traitant que les 25% de patchs visibles.
- Décodeur léger avec réinsertion des tokens [MASK] apprenables et unshuffle vectorisé.
- Normalisation des patchs de pixels (norm_pix_loss) et calcul de perte MSE masquée.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch.models.patch_embed import PatchEmbedding
from pytorch.models.block import TransformerBlock


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """Génère un encodage de position sin-cos 1D."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000.0 ** omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)
    return torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> torch.Tensor:
    """Génère l'encodage de position sin-cos 2D standard de MAE."""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # w d'abord puis h
    grid = torch.stack(grid, dim=0)  # (2, grid_size, grid_size)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed.unsqueeze(0)  # (1, N or 1+N, D)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0
    # Utilise la moitié des dimensions pour h, l'autre moitié pour w
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)


def patchify(imgs: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Découpe une image en patchs de pixels aplatis.
    Entrée  : imgs (B, 3, H, W)
    Sortie  : patchs (B, N, patch_size * patch_size * 3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0, (
        f"L'image ({imgs.shape[2]}x{imgs.shape[3]}) doit être divisible par patch_size={p}"
    )
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p * p * 3))
    return x


def unpatchify(x: torch.Tensor, patch_size: int = 16, img_size: int = 224) -> torch.Tensor:
    """
    Reconstitue une image RVB complète à partir de patchs de pixels aplatis.
    Entrée  : x (B, N, patch_size * patch_size * 3)
    Sortie  : imgs (B, 3, H, W)
    """
    p = patch_size
    h = w = img_size // p
    assert h * w == x.shape[1], f"Nombre de patchs {x.shape[1]} incompatible avec grille {h}x{w}"
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
    return imgs


class MaskedAutoencoderViT(nn.Module):
    """
    Vision Transformer Auto-supervisé par Masquage (Masked Autoencoder - MAE).
    Architecture asymétrique officielle de He et al. (CVPR 2022).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        self.norm_pix_loss = norm_pix_loss

        # ---------------------------------------------------------------------
        # 1. ENCODEUR MAE (ViT Lourd)
        # ---------------------------------------------------------------------
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_channels=in_chans, embed_dim=embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )  # sin-cos fixe non apprenable par défaut selon He et al.

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ---------------------------------------------------------------------
        # 2. DÉCODEUR MAE (Transformer Léger)
        # ---------------------------------------------------------------------
        # Projection linéaire de la dimension encodeur -> décodeur
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # Token [MASK] apprenable partagé pour tous les patchs masqués
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Tête de prédiction finale vers les pixels aplatis du patch
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size * patch_size * in_chans, bias=True
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialisation des poids selon le papier He et al. (CVPR 2022)."""
        # Initialisation sin-cos des encodages de position
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(pos_embed.float())

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.grid_size, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float())

        # Initialisation de la projection de patch similaire à nn.Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialisation du cls_token et mask_token
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # Initialisation des couches linéaires et LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Algorithme de masquage aléatoire par double argsort (Étape 2.1).
        
        Args:
            x: Tenseur de patchs (B, N, D)
            mask_ratio: Proportion de patchs à masquer (ex: 0.75 pour 75%)
            
        Returns:
            x_masked: Patchs visibles conservés (B, len_keep, D)
            mask: Masque binaire spatial (B, N) où 0 = conservé, 1 = masqué
            ids_restore: Indices pour réordonner les patchs à leur position originale
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        # 1. Bruit aléatoire uniforme
        noise = torch.rand(B, N, device=x.device)

        # 2. Premier argsort : mélange aléatoire
        ids_shuffle = torch.argsort(noise, dim=1)
        # 3. Second argsort : indices de restauration
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 4. Sélection des premiers len_keep patchs
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 5. Génération du masque binaire spatial (0: visible, 1: masqué)
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Passe avant de l'encodeur ViT (Étape 2.2) :
        Traite UNIQUEMENT les 25% de patchs visibles.
        """
        # Projection en patchs
        x = self.patch_embed(x)  # (B, N, D)

        # Ajout du position embedding (sans le CLS)
        x = x + self.pos_embed[:, 1:, :]

        # Masquage aléatoire
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Concaténation du [CLS] token avec son positional embedding
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + len_keep, D)

        # Passage dans les 12 blocs Transformer
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du décodeur léger (Étapes 2.3 & 2.4) :
        Réinsère les tokens [MASK] apprenables, unshuffle et reconstruit les pixels.
        """
        # 1. Projection vers la dimension du décodeur
        x = self.decoder_embed(x)

        # 2. Ajout des tokens [MASK] pour combler les positions manquantes
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        # x[:, 1:, :] sont les patchs visibles sans le CLS token
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # (B, N, D_dec)

        # 3. Unshuffle : remise en place spatiale exacte des 196 patchs
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        # 4. Réinsertion du [CLS] token au début
        x = torch.cat([x[:, :1, :], x_], dim=1)  # (B, 1 + N, D_dec)

        # 5. Ajout de l'encodage de position du décodeur
        x = x + self.decoder_pos_embed

        # 6. Blocs Transformer du décodeur
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # 7. Projection vers les pixels et retrait du [CLS] token
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # (B, N, patch_size * patch_size * 3)

        return x

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcul de la perte MSE masquée avec normalisation des patchs (Étapes 2.5 & 2.6).
        
        Args:
            imgs: Images originales (B, 3, H, W)
            pred: Patchs de pixels reconstruits (B, N, 768)
            mask: Masque binaire (B, N) où 1 = masqué, 0 = visible
        """
        target = patchify(imgs, patch_size=self.patch_size)

        if self.norm_pix_loss:
            # Normalisation individuelle de chaque patch par sa moyenne et sa variance
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        # Erreur quadratique par patch
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N) : moyenne sur les 768 pixels du patch

        # Perte calculée UNIQUEMENT sur les patchs masqués
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(
        self, imgs: torch.Tensor, mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Passe complète MAE : forward_encoder -> forward_decoder -> forward_loss.
        
        Returns:
            loss: Scalaire d'erreur de reconstruction masquée
            pred: Prédiction des pixels (B, N, patch_size**2 * 3)
            mask: Masque binaire spatial (B, N)
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# -----------------------------------------------------------------------------
# Fonctions Fabriques (Model Factories)
# -----------------------------------------------------------------------------

def mae_vit_base_patch16_224(**kwargs) -> MaskedAutoencoderViT:
    """Modèle MAE ViT-B/16 canonique (He et al., CVPR 2022)."""
    model = MaskedAutoencoderViT(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_pix_loss=True,
        **kwargs,
    )
    return model


def mae_vit_tiny_patch16_224(**kwargs) -> MaskedAutoencoderViT:
    """Modèle MAE ViT-Tiny ultra-léger pour itération rapide sur machine locale."""
    model = MaskedAutoencoderViT(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_num_heads=4,
        mlp_ratio=4.0,
        norm_pix_loss=True,
        **kwargs,
    )
    return model
