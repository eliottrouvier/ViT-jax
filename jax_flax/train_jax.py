"""
ViT-JAX — Phase 3 : Entraînement Fonctionnel JAX & Flax Linen (Étape 3.6)
Boucle d'entraînement pure compilée via @jax.jit avec l'optimiseur Optax AdamW.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import torch

from jax_flax.models.vit import VisionTransformerFlax

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_cifar10_arrays():
    """Charge les données CIFAR-10 en tableaux NumPy / JAX."""
    packed_file = DATA_DIR / "cifar10_packed.pt"
    if packed_file.exists():
        print("Chargement rapide des données depuis data/cifar10_packed.pt...")
        data = torch.load(packed_file, weights_only=True)
        # Normalisation [0, 255] -> [0.0, 1.0]
        train_imgs = data["train_imgs"].float().numpy() / 255.0
        train_labels = data["train_targets"].numpy()
        test_imgs = data["test_imgs"].float().numpy() / 255.0
        test_labels = data["test_targets"].numpy()
    else:
        print("Génération de données synthétiques pour CIFAR-10...")
        train_imgs = np.random.rand(1000, 3, 32, 32).astype(np.float32)
        train_labels = np.random.randint(0, 10, size=(1000,)).astype(np.int32)
        test_imgs = np.random.rand(200, 3, 32, 32).astype(np.float32)
        test_labels = np.random.randint(0, 10, size=(200,)).astype(np.int32)

    return train_imgs, train_labels, test_imgs, test_labels


def create_train_state(rng, model: VisionTransformerFlax, learning_rate: float, num_classes: int) -> train_state.TrainState:
    """Initialise le modèle Flax et crée l'état fonctionnel TrainState avec Optax."""
    dummy_input = jnp.zeros((1, 3, model.img_size, model.img_size), dtype=jnp.float32)
    variables = model.init(rng, dummy_input)
    params = variables["params"]

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


@jax.jit
def train_step(state: train_state.TrainState, images: jnp.ndarray, labels: jnp.ndarray):
    """Passe avant-arrière compilée par XLA (@jax.jit)."""
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, loss, acc


@jax.jit
def eval_step(state: train_state.TrainState, images: jnp.ndarray, labels: jnp.ndarray):
    """Évaluation compilée par XLA (@jax.jit)."""
    logits = state.apply_fn({"params": state.params}, images)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss, acc


def main():
    parser = argparse.ArgumentParser(description="Entraînement fonctionnel ViT en JAX & Flax")
    parser.add_argument("--epochs", type=int, default=5, help="Nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=128, help="Taille de batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taux d'apprentissage")
    parser.add_argument("--img-size", type=int, default=32, help="Résolution de l'image (32 pour CIFAR-10)")
    parser.add_argument("--num-classes", type=int, default=10, help="Nombre de classes")
    parser.add_argument("--max-batches", type=int, default=None, help="Limite de batchs")
    args = parser.parse_args()

    print("=" * 70)
    print("  ViT-JAX — Phase 3 : Entraînement Compilé JAX & Flax Linen (@jax.jit)")
    print("=" * 70)
    print(f"• Périphériques JAX   : {jax.devices()}")
    print(f"• Backend XLA         : {jax.default_backend()}")
    print(f"• Résolution d'image  : {args.img_size}x{args.img_size}")
    print(f"• Batch Size          : {args.batch_size}")
    print(f"• Learning Rate       : {args.lr}")

    train_imgs, train_labels, test_imgs, test_labels = load_cifar10_arrays()
    num_samples = len(train_imgs)
    steps_per_epoch = num_samples // args.batch_size

    # Instanciation du modèle ViT-Tiny Flax adapté pour CIFAR-10 (img_size=32, patch_size=4 ou 8)
    # Pour img_size=32, patch_size=4 donne 8x8 = 64 patchs
    model = VisionTransformerFlax(
        img_size=args.img_size,
        patch_size=4,
        in_channels=3,
        num_classes=args.num_classes,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
    )

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model, learning_rate=args.lr, num_classes=args.num_classes)

    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"• Paramètres du modèle: {total_params / 1e6:.2f} M")

    print("\n🚀 Démarrage de l'entraînement fonctionnel compilé (@jax.jit)...")
    train_losses, train_accs = [], []

    for epoch in range(1, args.epochs + 1):
        # Permutation aléatoire des indices
        rng, perm_rng = jax.random.split(rng)
        perm = np.random.permutation(num_samples)
        train_imgs_perm = train_imgs[perm]
        train_labels_perm = train_labels[perm]

        epoch_loss, epoch_acc, total_b = 0.0, 0.0, 0
        start_t = time.time()

        for step in range(steps_per_epoch):
            if args.max_batches and step >= args.max_batches:
                break

            b_imgs = jnp.array(train_imgs_perm[step * args.batch_size : (step + 1) * args.batch_size])
            b_lbls = jnp.array(train_labels_perm[step * args.batch_size : (step + 1) * args.batch_size])

            state, loss, acc = train_step(state, b_imgs, b_lbls)
            epoch_loss += float(loss)
            epoch_acc += float(acc)
            total_b += 1

        elapsed = time.time() - start_t
        epoch_loss /= max(1, total_b)
        epoch_acc /= max(1, total_b)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Validation rapide sur un lot de test
        val_imgs = jnp.array(test_imgs[:256])
        val_lbls = jnp.array(test_labels[:256])
        val_loss, val_acc = eval_step(state, val_imgs, val_lbls)

        print(f"Époque [{epoch}/{args.epochs}] ({elapsed:.1f}s) | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}% | Val Acc: {float(val_acc) * 100:.2f}%")

    print("\n✅ Entraînement JAX & Flax achevé avec succès !")


if __name__ == "__main__":
    main()
