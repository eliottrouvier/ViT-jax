"""
ViT-JAX — Phase 2B : Auto-Supervision par DINO (Self-Distillation)
Référence : Mathilde Caron et al., ICCV 2021 (papiers/dino.pdf)
"Emerging Properties in Self-Supervised Vision Transformers"

Implémentation complète de DINO :
- Pipeline Multi-Crop (2 vues globales + N vues locales).
- Tête de projection DINO (MLP 3 couches + normalisation L2 + prototypes).
- Mise à jour EMA du Professeur (Teacher) sans gradient.
- Centrage (Centering) & Aiguisement (Sharpening avec température).
- Perte Cross-Entropy auto-supervisée asymétrique.
- Extraction des cartes d'attention du [CLS] token.
"""

from typing import List, Tuple
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from pytorch.models.vit import VisionTransformer, vit_tiny_patch16_224, vit_base_patch16_224


# -----------------------------------------------------------------------------
# 1. Pipeline de Transformation Multi-Crop (Étape 2.8)
# -----------------------------------------------------------------------------

class DataAugmentationDINO:
    """
    Génère 2 vues globales (grand champ de vision) et N vues locales (détails).
    Adapté aussi bien pour la résolution 64x64 que 224x224.
    """
    def __init__(
        self,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.1, 0.4),
        local_crops_number: int = 4,
        global_size: int = 64,
        local_size: int = 32,
    ):
        self.local_crops_number = local_crops_number

        # Normalisation standard CIFAR / STL / ImageNet
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )

        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        clamp = transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))

        # 1. Première transformation globale (avec flip horizontal + color jitter)
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            clamp,
            transforms.RandomHorizontalFlip(p=0.5),
            color_jitter,
            normalize,
        ])

        # 2. Deuxième transformation globale
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            clamp,
            transforms.RandomHorizontalFlip(p=0.5),
            color_jitter,
            normalize,
        ])

        # 3. Transformations locales (petits zooms sur des détails)
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            clamp,
            transforms.RandomHorizontalFlip(p=0.5),
            color_jitter,
            normalize,
        ])

    def __call__(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        image: Tenseur (3, H, W) normalisé entre 0.0 et 1.0.
        Retourne une liste de 2 + local_crops_number vues.
        """
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


# -----------------------------------------------------------------------------
# 2. Tête de Projection DINO (Étape 2.11)
# -----------------------------------------------------------------------------

class DINOHead(nn.Module):
    """
    Tête de projection non linéaire :
    MLP 3 couches -> Normalisation L2 -> Prototypes normalisés en poids (sans biais).
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1024,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        nlayers: int = 3,
        hidden_dim: int = 512,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        # Couche de prototypes sans biais avec normalisation de poids
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant : MLP -> L2 Norm -> Prototypes."""
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


# -----------------------------------------------------------------------------
# 3. Fonction de Perte DINO : Centering & Sharpening (Étapes 2.12 & 2.13)
# -----------------------------------------------------------------------------

class DINOLoss(nn.Module):
    """
    Calcule la Cross-Entropy auto-supervisée entre l'Élève et le Professeur.
    Applique le Centrage (Centering) et l'Aiguisement (Sharpening avec température).
    """
    def __init__(
        self,
        out_dim: int,
        ncrops: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 0,
        nepochs: int = 10,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Schedule de température du professeur
        self.teacher_temp_schedule = torch.linspace(
            warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
        ).tolist() if warmup_teacher_temp_epochs > 0 else []
        self.teacher_temp_schedule += [teacher_temp] * (nepochs - len(self.teacher_temp_schedule))

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        """
        Cross-entropy entre les prédictions du Teacher (2 vues globales)
        et les prédictions du Student (toutes les vues).
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # Centrage et Sharpening sur le Teacher
        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # 2 vues globales uniquement

        total_loss = 0.0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Ne pas calculer la perte d'une vue globale contre elle-même
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """Mise à jour par moyenne mobile du centre pour éviter l'effondrement."""
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        # Moyenne mobile exponentielle du centre
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# -----------------------------------------------------------------------------
# 4. Modèle Wrapper DINO & Mise à Jour EMA (Étapes 2.9 & 2.10)
# -----------------------------------------------------------------------------

def update_teacher_ema(student: nn.Module, teacher: nn.Module, momentum: float = 0.996):
    """Mise à jour par moyenne mobile exponentielle (EMA) des poids du Teacher."""
    with torch.no_grad():
        for param_s, param_t in zip(student.parameters(), teacher.parameters()):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1.0 - momentum)


class DINOModel(nn.Module):
    """
    Assemble le backbone VisionTransformer et la tête de projection DINOHead.
    """
    def __init__(self, backbone: VisionTransformer, head: DINOHead):
        super().__init__()
        self.backbone = backbone
        # On supprime la tête de classification supervisée du ViT
        self.backbone.head = nn.Identity()
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extrait la représentation du CLS token puis projette via la DINOHead."""
        features = self.backbone(x)  # (B, embed_dim)
        logits = self.head(features)  # (B, out_dim)
        return logits

    @torch.no_grad()
    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Étape 2.14 : Extrait la carte d'attention de la dernière couche Transformer.
        Retourne : (B, num_heads, 197, 197) où [:, :, 0, 1:] est l'attention du [CLS] token.
        """
        B, C, H, W = x.shape
        x = self.backbone.patch_embed(x)
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.backbone.interpolate_pos_encoding(x, H, W)

        for i, blk in enumerate(self.backbone.blocks):
            if i < len(self.backbone.blocks) - 1:
                x = blk(x)
            else:
                # Dernière couche : on demande le retour de la matrice d'attention Softmax
                x, attn = blk(x, return_attention=True)
                return attn


# -----------------------------------------------------------------------------
# Fonctions Fabriques (Factories)
# -----------------------------------------------------------------------------

def dino_vit_tiny(out_dim: int = 1024, img_size: int = 64, patch_size: int = 16) -> Tuple[DINOModel, DINOModel]:
    """Instancie la paire Élève et Professeur ViT-Tiny pour DINO."""
    # 1. Élève (Student)
    backbone_s = vit_tiny_patch16_224(img_size=img_size, patch_size=patch_size, num_classes=0)
    head_s = DINOHead(in_dim=192, out_dim=out_dim, hidden_dim=512, bottleneck_dim=128)
    student = DINOModel(backbone_s, head_s)

    # 2. Professeur (Teacher) - Même architecture avec gradients gelés
    backbone_t = vit_tiny_patch16_224(img_size=img_size, patch_size=patch_size, num_classes=0)
    head_t = DINOHead(in_dim=192, out_dim=out_dim, hidden_dim=512, bottleneck_dim=128)
    teacher = DINOModel(backbone_t, head_t)

    # Synchronisation initiale stricte des poids
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher


def dino_vit_base(out_dim: int = 4096, img_size: int = 224, patch_size: int = 16) -> Tuple[DINOModel, DINOModel]:
    """Instancie la paire Élève et Professeur ViT-B/16 canonique pour DINO."""
    backbone_s = vit_base_patch16_224(img_size=img_size, patch_size=patch_size, num_classes=0)
    head_s = DINOHead(in_dim=768, out_dim=out_dim, hidden_dim=2048, bottleneck_dim=256)
    student = DINOModel(backbone_s, head_s)

    backbone_t = vit_base_patch16_224(img_size=img_size, patch_size=patch_size, num_classes=0)
    head_t = DINOHead(in_dim=768, out_dim=out_dim, hidden_dim=2048, bottleneck_dim=256)
    teacher = DINOModel(backbone_t, head_t)

    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher
