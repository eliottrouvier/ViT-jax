# 🧠 Fiche Théorique 01 : Patch Embedding & Encodage de Position

> **Module d'apprentissage ViT-JAX**  
> *Auteur : Eliott Rouvier*  
> *Référence : Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ICLR 2021)*  
> 📄 **Papier Original (PDF) :** [`papiers/vit.pdf`](../papiers/vit.pdf) · [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

---

## 🧭 Sommaire

1. [De l'Image 2D à la Séquence de Tokens 1D](#1-de-limage-2d-à-la-séquence-de-tokens-1d)
2. [Équivalence Mathématique entre Découpage et Conv2d](#2-équivalence-mathématique-entre-découpage-et-conv2d)
3. [Arithmétique des Paramètres : Poids, Biais et Données](#3-arithmétique-des-paramètres--poids-biais-et-données)
4. [Le Piège de la Disposition Mémoire (Row-Major) en PyTorch](#4-le-piège-de-la-disposition-mémoire-row-major-en-pytorch)
5. [Le Token [CLS] et le Positional Embedding (Le fameux 197)](#5-le-token-cls-et-le-positional-embedding-le-fameux-197)
6. [Résolution Dynamique & Rééchantillonnage 2D (Bicubique)](#6-résolution-dynamique--rééchantillonnage-2d-bicubique)
7. [Biais Inductif : Pourquoi ViT est "Affamé de Données" (Data Hunger)](#7-biais-inductif--pourquoi-vit-est-affamé-de-données-data-hunger)

---

## 1. De l'Image 2D à la Séquence de Tokens 1D

Un réseau convolutif classique (ResNet) traite une image comme un volume continu de pixels $\mathbb{R}^{H \times W \times C}$.  
En revanche, l'architecture **Transformer** standard a été conçue pour le texte : elle exige en entrée une **séquence 1D ordonnée de tokens** $\mathbb{R}^{N \times D}$.

### La discrétisation en patchs :
Pour une image standard de résolution $224 \times 224$ avec 3 canaux (RGB) :
- On fixe une taille de patch $P = 16$ pixels.
- Le nombre de patchs formés sur la grille est :
  $$N = \left(\frac{H}{P}\right) \times \left(\frac{W}{P}\right) = \left(\frac{224}{16}\right) \times \left(\frac{224}{16}\right) = 14 \times 14 = \mathbf{196 \text{ patchs}}$$
- Chaque patch individuel est un petit cube de pixels de dimensions :
  $$P \times P \times C = 16 \times 16 \times 3 = \mathbf{768 \text{ pixels}}$$

---

## 2. Équivalence Mathématique entre Découpage et Conv2d

Pour projeter chaque patch aplati de 768 pixels vers l'espace latent du modèle $D = 768$, l'opération mathématique théorique est une projection linéaire :
$$z_i = x_i \cdot W + b$$
avec $W \in \mathbb{R}^{768 \times 768}$ et $b \in \mathbb{R}^{768}$.

### Pourquoi `nn.Conv2d` réalise cette opération en 1 étape optimale :
Une couche convolutive 2D :
```python
nn.Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16)
```
- **`kernel_size=16`** : La fenêtre de convolution fait exactement la taille d'un patch ($16 \times 16$).
- **`stride=16`** : Le pas de déplacement est égal à la taille de la fenêtre $\implies$ **aucun chevauchement entre les patchs**.
- **`in_channels=3`** : Traite les 3 canaux de couleur.
- **`out_channels=768`** : Projette directement vers la dimension d'encodage $D = 768$.

La formule de la dimension spatiale de sortie de convolution confirme la grille :
$$H_{\text{out}} = \left\lfloor \frac{224 - 16}{16} \right\rfloor + 1 = 14, \qquad W_{\text{out}} = \left\lfloor \frac{224 - 16}{16} \right\rfloor + 1 = 14$$

Le tenseur en sortie de la convolution a donc la forme :
$$\mathbf{(B, 768, 14, 14)}$$

---

## 3. Arithmétique des Paramètres : Poids, Biais et Données

> [!IMPORTANT]
> **Distinction fondamentale : Données vs Paramètres**
> - **Données d'entrée :** $224 \times 224 \times 3 = 150\,528$ pixels par image. Ce ne sont **PAS** des paramètres du réseau (elles changent à chaque image).
> - **Paramètres apprenables :** Ce sont les poids stockés dans le modèle que la rétropropagation ajuste pendant l'apprentissage.

### Décompte exact des paramètres de `PatchEmbedding` :
1. **Les Poids ($W$) :**  
   Chaque filtre de sortie est de taille $(3, 16, 16) = 768$. Il y a 768 filtres distincts :
   $$W = 768 \times (3 \times 16 \times 16) = 768 \times 768 = \mathbf{589\,824 \text{ poids}}$$
2. **Le Biais ($b$) :**  
   Il y a exactement 1 biais par canal de sortie :
   $$b = \mathbf{768 \text{ biais}}$$
3. **Total des paramètres :**
   $$\text{Total} = 589\,824 + 768 = \mathbf{590\,592 \text{ paramètres}}$$

### Propriété d'indépendance de résolution :
Ce nombre de paramètres est **strictement invariant** par rapport à la taille de l'image. Que l'on passe une image $32 \times 32$ ou $1024 \times 1024$, les filtres restent identiques grâce au **partage de poids (Weight Sharing)**.

---

## 4. Le Piège de la Disposition Mémoire (Row-Major) en PyTorch

Le Transformer attend un tenseur de forme :
$$\mathbf{(B, N, D)} = (B, 196, 768)$$
Or, la convolution a produit :
$$\mathbf{(B, 768, 14, 14)}$$

### Le piège de `x.view(B, 196, 768)` :
En PyTorch, la mémoire physique est stockée de manière contiguë en ligne (*Row-Major order*).
Pour un tenseur `(B, 768, 14, 14)`, les données en mémoire sont ordonnées ainsi :
```
[Canal 0: 196 pixels de la grille], [Canal 1: 196 pixels], ..., [Canal 767: 196 pixels]
```
Si vous faites `view(B, 196, 768)`, PyTorch prend les 768 premiers flottants en mémoire pour former le premier token :
- Il prend les 196 pixels du canal 0, puis les 196 du canal 1, les 196 du canal 2, et une partie du canal 3 !
- **Désastre :** Les canaux et les positions spatiales sont entièrement mélangés. Les patchs perdent toute cohérence spatiale.

### La solution exacte :
Il faut obligatoirement aplatir d'abord la grille spatiale ($14 \times 14 \to 196$), puis échanger les axes :

```python
# 1. Aplatir uniquement la grille 2D spatiale : (B, 768, 14, 14) -> (B, 768, 196)
x = x.flatten(2)

# 2. Transposer les dimensions 1 (canaux) et 2 (tokens) : (B, 768, 196) -> (B, 196, 768)
x = x.transpose(1, 2)
```

Ou en une seule ligne avec `einops` :
```python
from einops import rearrange
x = rearrange(x, 'b c h w -> b (h w) c')  # (B, 196, 768)
```

---

## 5. Le Token [CLS] et le Positional Embedding (Le fameux 197)

```
Séquence d'entrée :
   ┌───────┬──────────┬──────────┬─────┬──────────┐
   │ [CLS] │ Patch 1  │ Patch 2  │ ... │Patch 196 │   = 197 tokens de dim 768
   └───────┴──────────┴──────────┴─────┴──────────┘
       +       +          +               +
   ┌───────┬──────────┬──────────┬─────┬──────────┐
   │ Pos 0 │  Pos 1   │  Pos 2   │ ... │ Pos 196  │   = pos_embed (1, 197, 768)
   └───────┴──────────┴──────────┴─────┴──────────┘
```

### 1. Pourquoi le `[CLS]` token ?
Le mécanisme de Self-Attention permet aux 196 patchs d'interagir entre eux. Mais pour la classification globale de l'image, aucun patch particulier (ex: le coin d'un ciel) ne doit porter la décision finale.  
On insère donc un **vecteur apprenable neutre `[CLS]`** à l'indice 0. Au fil des 12 couches de Transformer, le token `[CLS]` agrège l'information de tous les patchs pertinents. Sa représentation finale est ensuite envoyée au classifieur linéaire.

### 2. Pourquoi le `PositionalEmbedding` ?
Le mécanisme de Self-Attention est **invariant par permutation** : si vous mélangez les tokens dans le désordre, le calcul d'attention donne exactement le même résultat sans positional encoding.  
Pour préserver la géométrie spatiale, on ajoute un vecteur de position apprenable :
$$\mathbf{pos\_embed} \in \mathbb{R}^{1 \times 197 \times 768}$$
- Position 0 : correspond au `[CLS]` token.
- Positions 1 à 196 : correspondent aux 196 patchs dans l'ordre de lecture (gauche à droite, haut en bas).

---

## 6. Résolution Dynamique & Rééchantillonnage 2D (Bicubique)

### Le problème :
Si vous testez une image haute résolution $384 \times 384$ avec des patchs de 16 :
$$N_{\text{new}} = \left(\frac{384}{16}\right)^2 = 24 \times 24 = 576 \text{ patchs} \implies 577 \text{ tokens (avec CLS)}$$
Tenter d'ajouter un `pos_embed` pré-entraîné de taille 197 à 577 tokens provoque un **crash dimensionnel**.

### Terminologie du Rééchantillonnage (*Resampling*) :
- **Sur-échantillonnage (*Upsampling / Upscaling*) :** Agrandir la grille ($14 \times 14 \to 24 \times 24$, de 196 à 576 patchs).
- **Sous-échantillonnage (*Downsampling / Downscaling*) :** Réduire la grille ($14 \times 14 \to 7 \times 7$, de 196 à 49 patchs pour basse résolution type $112 \times 112$).

### Pourquoi l'interpolation Bi-cubique ?
- **Linéaire (1D) :** Relie deux points par une droite rigide (crée des angles pointus aux jonctions).
- **Cubique (1D) :** Relie les points par une courbe douce (polynôme de degré 3) en observant les 4 points voisins les plus proches.
- **Bi-cubique (2D) :** Applique cette interpolation cubique dans les **deux dimensions de l'espace ($X$ et $Y$)** en observant les **$4 \times 4 = 16$ points voisins**. C'est le standard de zoom doux d'images (comme dans Photoshop).

### Algorithme de redimensionnement de `pos_embed` :
```python
import torch
import torch.nn.functional as F

# pos_embed entraîné en 224x224 : (1, 197, 768)
pos_embed_trained = torch.randn(1, 197, 768)

# 1. Séparer le [CLS] token (fixe) des positions spatiales
cls_pos = pos_embed_trained[:, :1, :]      # (1, 1, 768)
patch_pos = pos_embed_trained[:, 1:, :]    # (1, 196, 768)

# 2. Déplier en grille 2D : (1, 768, 14, 14)
patch_pos_grid = patch_pos.reshape(1, 14, 14, 768).permute(0, 3, 1, 2)

# 3. Interpolation bicubique 2D vers la nouvelle résolution (ex: 24x24 ou 7x7)
new_grid = F.interpolate(
    patch_pos_grid, 
    size=(24, 24),            # Nouvelle taille voulue
    mode='bicubic', 
    align_corners=False,
    antialias=True            # Filtre anti-repliement (crucial en downsampling)
)

# 4. Ré-aplatir et recoller le [CLS] token
new_patch_pos = new_grid.permute(0, 2, 3, 1).flatten(1, 2) # (1, 576, 768)
new_pos_embed = torch.cat([cls_pos, new_patch_pos], dim=1)  # (1, 577, 768)
```

---

## 7. Biais Inductif : Pourquoi ViT est "Affamé de Données" (*Data Hunger*)

| Caractéristique | Convolution (CNN / ResNet) | Vision Transformer (ViT) |
| :--- | :--- | :--- |
| **Localité spatiale** | Forte : les filtres $3 \times 3$ ne regardent que les voisins immédiats. | Faible : uniquement à l'intérieur du patch $16 \times 16$. |
| **Invariance par translation** | Forte : intégrée par nature par le glissement des noyaux. | Faible : doit être apprise entièrement grâce aux données. |
| **Champ récepteur** | Croît lentement couche après couche. | **Global dès la couche 1** : chaque token peut regarder tous les autres. |
| **Besoin en données** | Faible à moyen (apprend vite sur petit dataset). | **Massif** : sans grand pré-entraînement (ImageNet-21k, JFT-300M), il surapprend. |

### Conclusion conceptuelle :
Parce que le Transformer n'a presque aucun biais inductif visuel pré-câblé, il ne fait aucune hypothèse sur la structure des images. Cela le rend plus difficile à entraîner sur de petits jeux de données, mais lui confère un **plafond de performance infiniment supérieur** lorsqu'il est alimenté par des volumes massifs de données (loi d'échelle / *scaling laws*).
