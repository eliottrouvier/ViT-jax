"""
Tests unitaires pour DINO (Self-Distillation) - Phase 2B
Vérifie le pipeline multi-crop, la tête DINO, la fonction de perte avec centrage/sharpening,
la mise à jour EMA du Professeur et l'extraction des cartes d'attention du [CLS] token.
"""

import pytest
import torch
import torch.nn.functional as F

from pytorch.self_supervised import (
    DataAugmentationDINO,
    DINOHead,
    DINOLoss,
    DINOModel,
    update_teacher_ema,
    dino_vit_tiny,
)


def test_data_augmentation_dino_multi_crop():
    """Vérifie que la transformation multi-crop génère 2 vues globales et 4 vues locales."""
    aug = DataAugmentationDINO(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.1, 0.4),
        local_crops_number=4,
        global_size=64,
        local_size=32,
    )
    img = torch.rand(3, 64, 64)
    crops = aug(img)

    assert len(crops) == 6, f"Attendu 6 crops (2 globaux + 4 locaux), obtenu {len(crops)}"
    # 2 vues globales (3, 64, 64)
    for i in range(2):
        assert crops[i].shape == (3, 64, 64), f"Vue globale {i} de forme inattendue : {crops[i].shape}"
    # 4 vues locales (3, 32, 32)
    for i in range(2, 6):
        assert crops[i].shape == (3, 32, 32), f"Vue locale {i} de forme inattendue : {crops[i].shape}"


def test_dino_head_projection():
    """Vérifie la dimension de sortie et l'absence de biais sur les prototypes."""
    head = DINOHead(in_dim=192, out_dim=1024, bottleneck_dim=128)
    x = torch.randn(4, 192)
    out = head(x)

    assert out.shape == (4, 1024), f"Forme de sortie inattendue : {out.shape}"
    # Vérification que la dernière couche est bien sans biais
    assert head.last_layer.bias is None


def test_dino_loss_and_centering():
    """Vérifie le calcul de la perte, l'asymétrie de température et la mise à jour du centre."""
    out_dim = 1024
    loss_fn = DINOLoss(
        out_dim=out_dim,
        ncrops=6,
        student_temp=0.1,
        teacher_temp=0.04,
        center_momentum=0.9,
    )

    B = 2
    # L'élève voit 6 vues concaténées en batch : 6 * B = 12
    student_output = torch.randn(6 * B, out_dim)
    # Le professeur ne voit que les 2 vues globales : 2 * B = 4
    teacher_output = torch.randn(2 * B, out_dim)

    initial_center = loss_fn.center.clone()
    assert torch.all(initial_center == 0.0)

    loss = loss_fn(student_output, teacher_output, epoch=0)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert loss.item() > 0.0

    # Vérification de la mise à jour du centre
    assert not torch.all(loss_fn.center == 0.0)
    assert loss_fn.center.shape == (1, out_dim)


def test_teacher_ema_update():
    """Vérifie que le Professeur est correctement mis à jour par EMA sans gradient."""
    student, teacher = dino_vit_tiny(out_dim=256, img_size=64)

    # Vérification que le prof ne requiert pas de gradient
    for p in teacher.parameters():
        assert not p.requires_grad

    # Perturbation artificielle des poids de l'élève
    with torch.no_grad():
        for p in student.parameters():
            p.add_(1.0)

    # Récupération d'un paramètre de référence
    t_param_old = next(teacher.parameters()).clone()
    s_param = next(student.parameters()).clone()

    momentum = 0.9
    update_teacher_ema(student, teacher, momentum=momentum)

    t_param_new = next(teacher.parameters())
    expected = momentum * t_param_old + (1.0 - momentum) * s_param

    assert torch.allclose(t_param_new, expected, atol=1e-6)


def test_get_last_selfattention():
    """Vérifie l'extraction de la matrice d'attention de la dernière couche."""
    student, _ = dino_vit_tiny(out_dim=256, img_size=64)
    student.eval()

    B = 2
    x = torch.randn(B, 3, 64, 64)
    attn = student.get_last_selfattention(x)

    # Avec img_size=64 et patch_size=16 : 4x4 = 16 patchs + 1 token CLS = 17 tokens
    expected_tokens = 17
    num_heads = 3  # ViT-Tiny possède 3 têtes
    assert attn.shape == (B, num_heads, expected_tokens, expected_tokens)

    # Vérification que chaque ligne de la matrice d'attention somme à 1 (propriété Softmax)
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_dino_training_step():
    """Vérifie une passe avant-arrière complète avec rétropropagation."""
    student, teacher = dino_vit_tiny(out_dim=512, img_size=64)
    loss_fn = DINOLoss(out_dim=512, ncrops=6)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    B = 2
    # 2 vues globales (64x64) et 4 vues locales (32x32)
    global_crops = [torch.randn(B, 3, 64, 64), torch.randn(B, 3, 64, 64)]
    local_crops = [torch.randn(B, 3, 32, 32) for _ in range(4)]

    # Professeur ne prend que les vues globales
    teacher_inputs = torch.cat(global_crops, dim=0)
    with torch.no_grad():
        teacher_output = teacher(teacher_inputs)

    # Élève prend toutes les vues (les locales sont traitées par interpolation positionnelle automatique)
    student_outputs = []
    for crop in global_crops + local_crops:
        student_outputs.append(student(crop))
    student_output = torch.cat(student_outputs, dim=0)

    loss = loss_fn(student_output, teacher_output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # EMA
    update_teacher_ema(student, teacher, momentum=0.99)
    assert loss.item() > 0.0
