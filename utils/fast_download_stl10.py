"""
ViT-JAX — Téléchargement Parallèle Accéléré de STL-10 (10 flux simultanés)
Télécharge stl10_binary.tar.gz par blocs de bytes via multi-threading,
décompresse et crée data/stl10_packed.pt en quelques minutes.
"""

import os
import sys
import time
import tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = DATA_DIR / "stl10_packed.pt"
TAR_FILE = DATA_DIR / "stl10_binary.tar.gz"

URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
TOTAL_SIZE = 2640397119
NUM_WORKERS = 10


def download_chunk(idx, start_byte, end_byte, chunk_path):
    """Télécharge un segment de bytes précis."""
    req = urllib.request.Request(URL)
    req.headers["Range"] = f"bytes={start_byte}-{end_byte}"
    with urllib.request.urlopen(req, timeout=60) as resp, open(chunk_path, "wb") as f:
        while True:
            buf = resp.read(1024 * 512)
            if not buf:
                break
            f.write(buf)
    return idx


def download_multithreaded():
    """Télécharge l'archive complète en parallèle avec N threads."""
    if TAR_FILE.exists() and TAR_FILE.stat().st_size == TOTAL_SIZE:
        print(f"✅ Archive déjà téléchargée : {TAR_FILE}")
        return

    chunk_size = TOTAL_SIZE // NUM_WORKERS
    chunks = []
    chunk_paths = []

    for i in range(NUM_WORKERS):
        start = i * chunk_size
        end = TOTAL_SIZE - 1 if i == NUM_WORKERS - 1 else (i + 1) * chunk_size - 1
        path = DATA_DIR / f"stl10_part_{i}.tmp"
        chunks.append((i, start, end, path))
        chunk_paths.append(path)

    print(f"🚀 Lancement du téléchargement parallèle ({NUM_WORKERS} flux simultanés)...")
    start_t = time.time()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(download_chunk, idx, start, end, path): idx
            for idx, start, end, path in chunks
        }
        completed = 0
        for future in as_completed(futures):
            idx = future.result()
            completed += 1
            elapsed = time.time() - start_t
            speed_mb = (completed * chunk_size) / (elapsed * 1024 * 1024)
            print(f"  👉 Bloc {completed}/{NUM_WORKERS} terminé ({speed_mb:.1f} Mo/s en moyenne)")

    print(f"🧩 Assemblage des {NUM_WORKERS} blocs dans {TAR_FILE.name}...")
    with open(TAR_FILE, "wb") as outfile:
        for path in chunk_paths:
            with open(path, "rb") as infile:
                while True:
                    data = infile.read(1024 * 1024 * 4)
                    if not data:
                        break
                    outfile.write(data)
            path.unlink()  # suppression du fichier temporaire

    print(f"✅ Téléchargement achevé en {time.time() - start_t:.1f}s !")


def extract_and_pack(num_train: int = 20000, num_test: int = 1000):
    """Extrait directement les tenseurs nécessaires et crée stl10_packed.pt."""
    if OUT_FILE.exists():
        print(f"✅ Dataset packed déjà prêt : {OUT_FILE}")
        return

    print("📦 Décompression et lecture directe depuis le tar.gz...")
    start_t = time.time()
    with tarfile.open(TAR_FILE, "r:gz") as tar:
        # 1. Lecture de test_X.bin pour les images d'évaluation
        print("  -> Lecture de stl10_binary/test_X.bin...")
        test_member = tar.getmember("stl10_binary/test_X.bin")
        with tar.extractfile(test_member) as f:
            raw_test = f.read(num_test * 96 * 96 * 3)
            # STL-10 stocke en ordre Fortran column-major (3, 96, 96)
            test_np = np.frombuffer(raw_test, dtype=np.uint8).reshape(num_test, 3, 96, 96)
            test_np = np.transpose(test_np, (0, 1, 3, 2))  # correction orientation column-major
            test_imgs = torch.from_numpy(test_np.copy())

        # 2. Lecture de unlabeled_X.bin pour l'auto-supervision
        print(f"  -> Lecture des {num_train} premières images de stl10_binary/unlabeled_X.bin...")
        unlabeled_member = tar.getmember("stl10_binary/unlabeled_X.bin")
        with tar.extractfile(unlabeled_member) as f:
            raw_train = f.read(num_train * 96 * 96 * 3)
            train_np = np.frombuffer(raw_train, dtype=np.uint8).reshape(num_train, 3, 96, 96)
            train_np = np.transpose(train_np, (0, 1, 3, 2))  # correction orientation column-major
            train_imgs = torch.from_numpy(train_np.copy())

    print(f"💾 Sauvegarde de {OUT_FILE}...")
    torch.save({
        "train_imgs": train_imgs,
        "test_imgs": test_imgs,
    }, OUT_FILE)

    mb = OUT_FILE.stat().st_size / (1024 * 1024)
    print(f"🎉 Terminé en {time.time() - start_t:.1f}s ! {OUT_FILE} généré ({mb:.1f} Mo).")


def main():
    if not OUT_FILE.exists():
        download_multithreaded()
        extract_and_pack(num_train=20000, num_test=1000)
    else:
        print(f"✅ {OUT_FILE} est déjà présent.")


if __name__ == "__main__":
    main()
