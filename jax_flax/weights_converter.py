"""
ViT-JAX — Phase 3 : Convertisseur de Poids PyTorch vers Flax Linen (Étape 3.4)
Transpose et restructure fidèlement les poids d'un state_dict PyTorch (.pt)
en un PyTree / FrozenDict compatible Flax Linen.

Règles de transposition :
- Linear PyTorch (out, in) -> Dense Flax (in, out) : W_flax = W_torch^T
- Conv2d PyTorch (C_out, C_in, kH, kW) -> Conv Flax (kH, kW, C_in, C_out) : permute(2, 3, 1, 0)
- LayerNorm PyTorch (weight, bias) -> LayerNorm Flax (scale, bias)
"""

from typing import Dict, Any, Union
from pathlib import Path
import numpy as np
import torch
from flax.core import freeze, FrozenDict

from pytorch.models.vit import VisionTransformer


def torch_to_flax_state_dict(
    torch_state: Union[Dict[str, torch.Tensor], VisionTransformer],
    num_layers: int = 12,
) -> FrozenDict:
    """
    Convertit un state_dict PyTorch ou un modèle nn.Module en FrozenDict Flax.
    """
    if isinstance(torch_state, torch.nn.Module):
        state_dict = torch_state.state_dict()
    else:
        state_dict = torch_state

    # Helper pour convertir tenseur Torch en numpy array float32
    def to_np(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().float().numpy()

    params: Dict[str, Any] = {}

    # 1. Patch Embedding
    patch_embed_params = {}
    if "patch_embed.proj.weight" in state_dict:
        # Conv2d : (C_out, C_in, kH, kW) -> (kH, kW, C_in, C_out)
        w = to_np(state_dict["patch_embed.proj.weight"])
        w_flax = np.transpose(w, (2, 3, 1, 0))
        proj_params = {"kernel": w_flax}
        if "patch_embed.proj.bias" in state_dict and state_dict["patch_embed.proj.bias"] is not None:
            proj_params["bias"] = to_np(state_dict["patch_embed.proj.bias"])
        patch_embed_params["proj"] = proj_params
    params["patch_embed"] = patch_embed_params

    # 2. CLS Token et Positional Embedding
    if "cls_token" in state_dict:
        params["cls_token"] = to_np(state_dict["cls_token"])
    if "pos_embed" in state_dict:
        params["pos_embed"] = to_np(state_dict["pos_embed"])

    # 3. Blocs Transformer
    for i in range(num_layers):
        blk_key = f"blocks_{i}"
        blk_params = {}

        # Norm 1 (Pre-LayerNorm Attention)
        n1_w = state_dict.get(f"blocks.{i}.norm1.weight")
        n1_b = state_dict.get(f"blocks.{i}.norm1.bias")
        if n1_w is not None and n1_b is not None:
            blk_params["norm1"] = {"scale": to_np(n1_w), "bias": to_np(n1_b)}

        # Attention Multi-Head (QKV + Proj)
        attn_params = {}
        qkv_w = state_dict.get(f"blocks.{i}.attn.qkv.weight")
        qkv_b = state_dict.get(f"blocks.{i}.attn.qkv.bias")
        if qkv_w is not None:
            # Linear : (out_features, in_features) -> (in_features, out_features)
            qkv_dict = {"kernel": np.transpose(to_np(qkv_w), (1, 0))}
            if qkv_b is not None:
                qkv_dict["bias"] = to_np(qkv_b)
            attn_params["qkv"] = qkv_dict

        proj_w = state_dict.get(f"blocks.{i}.attn.proj.weight")
        proj_b = state_dict.get(f"blocks.{i}.attn.proj.bias")
        if proj_w is not None:
            proj_dict = {"kernel": np.transpose(to_np(proj_w), (1, 0))}
            if proj_b is not None:
                proj_dict["bias"] = to_np(proj_b)
            attn_params["proj"] = proj_dict
        blk_params["attn"] = attn_params

        # Norm 2 (Pre-LayerNorm MLP)
        n2_w = state_dict.get(f"blocks.{i}.norm2.weight")
        n2_b = state_dict.get(f"blocks.{i}.norm2.bias")
        if n2_w is not None and n2_b is not None:
            blk_params["norm2"] = {"scale": to_np(n2_w), "bias": to_np(n2_b)}

        # MLP Block (fc1 + fc2)
        mlp_params = {}
        fc1_w = state_dict.get(f"blocks.{i}.mlp.fc1.weight")
        fc1_b = state_dict.get(f"blocks.{i}.mlp.fc1.bias")
        if fc1_w is not None:
            fc1_dict = {"kernel": np.transpose(to_np(fc1_w), (1, 0))}
            if fc1_b is not None:
                fc1_dict["bias"] = to_np(fc1_b)
            mlp_params["fc1"] = fc1_dict

        fc2_w = state_dict.get(f"blocks.{i}.mlp.fc2.weight")
        fc2_b = state_dict.get(f"blocks.{i}.mlp.fc2.bias")
        if fc2_w is not None:
            fc2_dict = {"kernel": np.transpose(to_np(fc2_w), (1, 0))}
            if fc2_b is not None:
                fc2_dict["bias"] = to_np(fc2_b)
            mlp_params["fc2"] = fc2_dict
        blk_params["mlp"] = mlp_params

        params[blk_key] = blk_params

    # 4. LayerNorm finale
    if "norm.weight" in state_dict and "norm.bias" in state_dict:
        params["norm"] = {
            "scale": to_np(state_dict["norm.weight"]),
            "bias": to_np(state_dict["norm.bias"]),
        }

    # 5. Tête de classification (Head)
    if "head.weight" in state_dict and state_dict["head.weight"] is not None:
        head_w = state_dict["head.weight"]
        head_params = {"kernel": np.transpose(to_np(head_w), (1, 0))}
        if "head.bias" in state_dict and state_dict["head.bias"] is not None:
            head_params["bias"] = to_np(state_dict["head.bias"])
        params["head"] = head_params

    return freeze({"params": params})


def convert_checkpoint_file(torch_ckpt_path: Union[str, Path], save_flax_path: Union[str, Path] = None):
    """Charge un fichier checkpoint PyTorch (.pt) et le convertit en dictionnaire Flax."""
    torch_ckpt_path = Path(torch_ckpt_path)
    state = torch.load(torch_ckpt_path, map_location="cpu", weights_only=True)
    if "teacher" in state:
        state = state["teacher"]
    elif "model" in state:
        state = state["model"]

    flax_params = torch_to_flax_state_dict(state)
    print(f"✅ Conversion réussie de {torch_ckpt_path.name} vers structure Flax !")
    return flax_params
