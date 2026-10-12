# 📚 Bibliothèque des Papiers Fondateurs (PDFs)

> Ce dossier rassemble les 6 publications scientifiques majeures qui constituent les fondations théoriques et algorithmiques du projet **ViT-JAX**.  
> Chaque fois qu'une méthode est mentionnée dans le code ou les fiches théoriques, le lien renvoie directement vers son fichier PDF ici.

---

## 🧭 Index des Publications

| # | Fichier PDF | Titre de l'Article | Auteurs & Conférence | Concept Clé dans ViT-JAX |
| :-: | :--- | :--- | :--- | :--- |
| **01** | [📄 `attention.pdf`](./attention.pdf) | **Attention Is All You Need** | Vaswani et al. *(NeurIPS 2017)* | Architecture Transformer originale, mécanisme d'attention multi-têtes ($Q, K, V$). |
| **02** | [📄 `vit.pdf`](./vit.pdf) | **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** | Dosovitskiy et al. *(ICLR 2021)* | Adaptation du Transformer à la vision : Patch Embedding 16×16, `[CLS]` token, Positional Encoding. |
| **03** | [📄 `deit.pdf`](./deit.pdf) | **Training data-efficient image transformers & distillation through attention** | Touvron et al. *(ICML 2021)* | Entraînement de ViT sans pré-entraînement massif (distillation d'attention, résout le *data hunger*). |
| **04** | [📄 `dino.pdf`](./dino.pdf) | **Emerging Properties in Self-Supervised Vision Transformers** | Caron et al. *(ICCV 2021)* | Auto-supervision sans étiquette par distillation Élève / Professeur (EMA), Centering, Sharpening. |
| **05** | [📄 `mae.pdf`](./mae.pdf) | **Masked Autoencoders Are Scalable Vision Learners** | He et al. *(CVPR 2022)* | Auto-supervision par reconstruction de puzzle : masquage de 75 % des patchs, encodeur asymétrique. |
| **06** | [📄 `dinov2.pdf`](./dinov2.pdf) | **DINOv2: Learning Robust Visual Features without Supervision** | Oquab et al. *(2023)* | Extension à grande échelle de DINO (modèles géants, représentations visuelles universelles). |

---

## 🔬 Résumés Flash par Phase du Projet

### Phase 1 : ViT-B/16 from Scratch
- **[`attention.pdf`](./attention.pdf) :** Comprendre l'équation $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$, la projection linéaire multi-têtes et le bloc MLP avec résidus Pre-LayerNorm.
- **[`vit.pdf`](./vit.pdf) :** Comprendre le découpage de l'image en 196 patchs, la projection par convolution $16 \times 16$, l'ajout du jeton `[CLS]` (197 tokens) et l'interpolation bicubique des positions pour résolutions dynamiques.
- **[`deit.pdf`](./deit.pdf) :** Comprendre les recettes modernes d'optimisation et d'augmentation de données nécessaires pour faire converger un ViT sans surapprentissage.

### Phase 2 : Auto-Supervision (SSL)
- **[`mae.pdf`](./mae.pdf) (Phase 2A) :** Pourquoi masquer 75 % des patchs (redondance d'information visuelle), pourquoi l'encodeur ne traite que les 25 % visibles, et comment calculer la perte MSE uniquement sur les patchs manquants.
- **[`dino.pdf`](./dino.pdf) & [`dinov2.pdf`](./dinov2.pdf) (Phase 2B) :** Comment deux modèles (Student et Teacher) apprennent sans s'effondrer vers une solution triviale grâce au Centering et au Sharpening à basse température, et comment les cartes d'attention du `[CLS]` token réalisent une segmentation sémantique non supervisée.
