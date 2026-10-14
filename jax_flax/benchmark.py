"""
ViT-JAX — Phase 3 : Benchmark Comparatif PyTorch vs JAX/Flax XLA (Étape 3.7)
Mesure la latence par passe avant et le débit (throughput en images/seconde)
entre PyTorch (MPS/CPU) et JAX/Flax avec compilation XLA (@jax.jit).
Génère le graphique comparatif : outputs/benchmark_pytorch_vs_jax.png
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

from pytorch.models.vit import VisionTransformer
from jax_flax.models.vit import VisionTransformerFlax

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_pytorch(
    batch_size: int = 32,
    img_size: int = 224,
    embed_dim: int = 192,
    depth: int = 12,
    num_heads: int = 3,
    num_iters: int = 40,
    warmup: int = 10,
):
    """Mesure la vitesse d'inférence et d'entraînement en PyTorch."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔥 PyTorch Benchmark sur {device} (batch_size={batch_size})...")

    model = VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_classes=1000,
    ).to(device)
    model.eval()

    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    target = torch.randint(0, 1000, (batch_size,), device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 1. Warmup Inférence
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device.type == "mps":
                torch.mps.synchronize()

    # 2. Mesure Inférence
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(x)
            if device.type == "mps":
                torch.mps.synchronize()
    inf_time = (time.time() - start) / num_iters
    inf_fps = batch_size / inf_time

    # 3. Mesure Entraînement (Forward + Backward + Step)
    model.train()
    for _ in range(warmup):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()

    start = time.time()
    for _ in range(num_iters):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
    train_time = (time.time() - start) / num_iters
    train_fps = batch_size / train_time

    print(f"  PyTorch Inférence : {inf_time * 1000:.2f} ms/lot | {inf_fps:.1f} img/s")
    print(f"  PyTorch Train     : {train_time * 1000:.2f} ms/lot | {train_fps:.1f} img/s")
    return {"inf_time_ms": inf_time * 1000, "inf_fps": inf_fps, "train_fps": train_fps}


def benchmark_jax_flax(
    batch_size: int = 32,
    img_size: int = 224,
    embed_dim: int = 192,
    depth: int = 12,
    num_heads: int = 3,
    num_iters: int = 40,
    warmup: int = 10,
):
    """Mesure la vitesse d'inférence et d'entraînement en JAX/Flax avec XLA (@jax.jit)."""
    print(f"⚡ JAX/Flax Benchmark avec XLA @jax.jit (batch_size={batch_size})...")

    model = VisionTransformerFlax(
        img_size=img_size,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_classes=1000,
    )

    rng = jax.random.PRNGKey(0)
    dummy_x = jnp.zeros((batch_size, 3, img_size, img_size))
    variables = model.init(rng, dummy_x)
    params = variables["params"]

    tx = optax.adamw(learning_rate=1e-4)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Inférence fonctionnelle compilée
    @jax.jit
    def jitted_infer(p, x):
        return model.apply({"params": p}, x)

    # Entraînement fonctionnel compilé (loss + grads + optax step)
    @jax.jit
    def jitted_train_step(st, x, y):
        def loss_fn(p):
            logits = st.apply_fn({"params": p}, x)
            one_hot = jax.nn.one_hot(y, 1000)
            return optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        grads = jax.grad(loss_fn)(st.params)
        return st.apply_gradients(grads=grads)

    x_jax = jax.random.normal(rng, (batch_size, 3, img_size, img_size))
    y_jax = jax.random.randint(rng, (batch_size,), 0, 1000)

    # 1. Warmup & Compilation XLA Inférence
    for _ in range(warmup):
        res = jitted_infer(state.params, x_jax)
        res.block_until_ready()

    # 2. Mesure Inférence
    start = time.time()
    for _ in range(num_iters):
        res = jitted_infer(state.params, x_jax)
        res.block_until_ready()
    inf_time = (time.time() - start) / num_iters
    inf_fps = batch_size / inf_time

    # 3. Warmup & Compilation XLA Entraînement
    for _ in range(warmup):
        state = jitted_train_step(state, x_jax, y_jax)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), state.params)

    # 4. Mesure Entraînement
    start = time.time()
    for _ in range(num_iters):
        state = jitted_train_step(state, x_jax, y_jax)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), state.params)
    train_time = (time.time() - start) / num_iters
    train_fps = batch_size / train_time

    print(f"  JAX/Flax Inférence : {inf_time * 1000:.2f} ms/lot | {inf_fps:.1f} img/s")
    print(f"  JAX/Flax Train     : {train_time * 1000:.2f} ms/lot | {train_fps:.1f} img/s")
    return {"inf_time_ms": inf_time * 1000, "inf_fps": inf_fps, "train_fps": train_fps}


def plot_benchmark(pt_metrics, jax_metrics, out_path):
    """Génère le graphique comparatif des performances."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    frameworks = ["PyTorch", "JAX / Flax (XLA)"]
    colors = ["#f97316", "#0284c7"]

    # Graphique 1 : Débit d'Inférence (FPS)
    inf_fps_vals = [pt_metrics["inf_fps"], jax_metrics["inf_fps"]]
    bars1 = axes[0].bar(frameworks, inf_fps_vals, color=colors, width=0.45)
    axes[0].set_title("Débit d'Inférence (Throughput)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Images / seconde (plus haut = meilleur)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f} img/s", ha="center", va="bottom", fontweight="bold")

    # Graphique 2 : Débit d'Entraînement (FPS)
    train_fps_vals = [pt_metrics["train_fps"], jax_metrics["train_fps"]]
    bars2 = axes[1].bar(frameworks, train_fps_vals, color=colors, width=0.45)
    axes[1].set_title("Débit d'Entraînement (Forward+Backward)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Images / seconde (plus haut = meilleur)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f} img/s", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Benchmark Comparatif : PyTorch vs JAX & Flax Linen (ViT-Tiny)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Graphique de benchmark sauvegardé : {out_path}")


def main():
    print("=" * 70)
    print("  ViT-JAX — Phase 3 : Benchmark Comparatif PyTorch vs JAX/Flax")
    print("=" * 70)

    pt_res = benchmark_pytorch(batch_size=32, img_size=64, num_iters=30, warmup=5)
    jax_res = benchmark_jax_flax(batch_size=32, img_size=64, num_iters=30, warmup=5)

    out_file = OUTPUTS_DIR / "benchmark_pytorch_vs_jax.png"
    plot_benchmark(pt_res, jax_res, out_file)
    print("\n✅ Benchmark terminé avec succès !")


if __name__ == "__main__":
    main()
