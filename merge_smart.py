#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
combine_neurons_pca_lora.py ────────────────────────────────────────────────
Merge **shared** neurons from two source models ("A" and "B") into a base
model using one of three strategies:

1. **mean** – element-wise arithmetic mean (legacy behaviour)
2. **pca**  – PCA/SVD fusion in a low-dimensional principal-component space
3. **lora** – low-rank (LoRA-style) fusion of full projection matrices

Also supports extended variants:
- lora_plus_single
- lora_plus_multi
- lora_stable

Now with **InternVL2 / InternVL2.5** compatibility via smart loader.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Set, Union, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoConfig

###############################################################################
# Smart loader (Qwen2.5-VL & InternVL2/2.5)                                   #
###############################################################################
from transformers import AutoConfig, AutoModelForCausalLM

def smart_load_model(path: str, dtype, device: str):
    """
    Smartly select the most appropriate `from_pretrained` class.

    Heuristics:
    - Qwen2.5-VL: Qwen2_5_VLForConditionalGeneration
    - InternVL 2 / 2.5: InternVL*ForConditionalGeneration / InternVLChatModel
    - LLaVA / LLaVA-Next: LlavaNextForConditionalGeneration or AutoModelForVision2Seq
    - Otherwise: AutoModelForCausalLM
    """
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)

    def _lower_list(x):
        return [str(s).lower() for s in (x or [])]

    archs = _lower_list(getattr(cfg, "architectures", []))
    model_type = (getattr(cfg, "model_type", "") or "").lower()
    cfg_cls = cfg.__class__.__name__.lower()

    # Qwen2.5-VL family
    if ("qwen2_5_vlforconditionalgeneration" in archs or
        "qwen2.5-vl" in model_type or
        "qwen2_vl" in model_type or
        "qwen2.5_vl" in model_type):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path, torch_dtype=dtype, device_map=device, trust_remote_code=True
            )
        except Exception:
            # Fallback to AutoModelForCausalLM below.
            pass

    # InternVL 2 / 2.5 family
    if ("internvl" in model_type or any("internvl" in a for a in archs)):
        for cls_name in [
            "InternVL2ForConditionalGeneration",
            "InternVLForConditionalGeneration",
            "InternVLChatModel",
        ]:
            try:
                mod = __import__("transformers", fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                return cls.from_pretrained(
                    path, torch_dtype=dtype, device_map=device, trust_remote_code=True
                )
            except Exception:
                continue
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, device_map=device, trust_remote_code=True
        )

    # LLaVA / LLaVA-Next family (e.g., LlavaNextConfig / LlavaNextForConditionalGeneration)
    if ("llava" in model_type or
        any("llava" in a for a in archs) or
        "llava" in cfg_cls):
        # Prefer the explicit class if available (works with stock Transformers that include it)
        try:
            from transformers import LlavaNextForConditionalGeneration
            return LlavaNextForConditionalGeneration.from_pretrained(
                path, torch_dtype=dtype, device_map=device, trust_remote_code=True
            )
        except Exception:
            # Generic VL fallback (relies on trust_remote_code if the repo provides a custom class)
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq.from_pretrained(
                path, torch_dtype=dtype, device_map=device, trust_remote_code=True
            )

    # Generic Vision-Language fallback for other VL families
    vl_indicators = [
        "vision", "vl", "idefics", "paligemma", "phi3v", "phi-3-vision", "mllama"
    ]
    if (any(k in model_type for k in vl_indicators) or
        any(any(k in a for k in vl_indicators) for a in archs)):
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq.from_pretrained(
            path, torch_dtype=dtype, device_map=device, trust_remote_code=True
        )

    # Default fallback
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, device_map=device, trust_remote_code=True
    )


###############################################################################
# Helper functions                                                            #
###############################################################################

def read_neuron(path: Union[str, Path]) -> Dict[str, Dict[str, Set[int]]]:
    """Load a neuron-spec JSON and convert index lists to *sets* for O(1) lookup."""
    with open(path, "r") as f:
        data = json.load(f)
    return {
        group: {layer: set(indices) for layer, indices in layers.items()}
        for group, layers in data.items()
    }

###############################################################################
# Fusion helpers                                                              #
###############################################################################

@torch.no_grad()
def merge_neurons_with_pca(
    W_base: torch.Tensor,
    W_a: torch.Tensor,
    W_b: torch.Tensor,
    neuron_indices: Iterable[int],
    index_dim: int,
    k: int = 10,
    alpha: float = 0.5,
    div: int = 1,
    clip_ratio: float = 1.0,   # 1.0 -> no clipping; <1 for more conservative
) -> torch.Tensor:
    """
    PCA-based low-rank fusion on residuals ΔA, ΔB (in-place on W_base).

    Returns
    -------
    torch.Tensor
        The same W_base handle (for API compatibility).
    """
    if not neuron_indices:
        return W_base

    device, dtype = W_base.device, W_base.dtype
    neuron_indices = list(neuron_indices)
    N = len(neuron_indices)

    # ---------- 1. Collect residual vectors (Delta = W_branch - W_base) ----------
    vecs: List[torch.Tensor] = []
    for idx in neuron_indices:
        if index_dim == 0:   # row (out-features)
            vecs.append((W_a[idx // div] - W_base[idx // div]).to(torch.float32))
        else:                # column (in-features)
            vecs.append((W_a[:, idx // div] - W_base[:, idx // div]).to(torch.float32))
    for idx in neuron_indices:
        if index_dim == 0:
            vecs.append((W_b[idx // div] - W_base[idx // div]).to(torch.float32))
        else:
            vecs.append((W_b[:, idx // div] - W_base[:, idx // div]).to(torch.float32))

    X = torch.stack(vecs, dim=0)        # shape (2N, d)
    mean = X.mean(0, keepdim=True)      # (1, d)
    Xc = X - mean

    # ---------- 2. PCA / truncated SVD ----------
    k_eff = min(k, Xc.shape[0], Xc.shape[1])
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)        # on Xc.device
    V_k = Vt[:k_eff]                                            # (k_eff, d)
    coeffs = Xc @ V_k.T                                         # (2N, k_eff)

    coeffs_a, coeffs_b = coeffs[:N], coeffs[N:]
    coeffs_m = alpha * coeffs_a + (1 - alpha) * coeffs_b        # (N, k_eff)

    recon = coeffs_m @ V_k + mean                               # (N, d)  -> Δ̂

    # ---------- 3. Write back (incremental + clipping) ----------
    eps = 1e-6
    for j, idx in enumerate(neuron_indices):
        delta_j = recon[j]
        if index_dim == 0:              # row
            base_vec = W_base[idx // div].to(torch.float32)
            if clip_ratio < 1.0:
                ratio = delta_j.norm() / (base_vec.norm() + eps)
                if ratio > clip_ratio:
                    delta_j = delta_j * (clip_ratio / ratio)
            W_base[idx // div] = (base_vec + delta_j).to(dtype)
        else:                           # column
            base_vec = W_base[:, idx // div].to(torch.float32)
            if clip_ratio < 1.0:
                ratio = delta_j.norm() / (base_vec.norm() + eps)
                if ratio > clip_ratio:
                    delta_j = delta_j * (clip_ratio / ratio)
            W_base[:, idx // div] = (base_vec + delta_j).to(dtype)

    return W_base


def _rank_trunc_svd(W: torch.Tensor, rank: int) -> torch.Tensor:
    """Utility: best rank-*r* approximation via truncated SVD."""
    if rank <= 0 or torch.count_nonzero(W) == 0:
        return torch.zeros_like(W)

    orig_dtype = W.dtype
    device = W.device

    W_fp32 = W.to(dtype=torch.float32)
    U, S, Vt = torch.linalg.svd(W_fp32, full_matrices=False)

    U_r = U[:, :rank]
    S_r = torch.diag(S[:rank])
    Vt_r = Vt[:rank, :]

    W_approx = U_r @ S_r @ Vt_r
    return W_approx.to(dtype=orig_dtype, device=device)


@torch.no_grad()
def merge_with_lora(
    W_base: torch.Tensor,
    W_a: torch.Tensor,
    W_b: torch.Tensor,
    neuron_indices: Iterable[int],
    index_dim: int,
    rank: int = 4,
    beta: float = 0.5,
    div : int = 1,
) -> torch.Tensor:
    """LoRA-style low-rank fusion.

    1.  Compute **full-matrix** low-rank deltas for A/Base and B/Base.
    2.  Blend the two deltas by β.
    3.  Write the blended update back **only** to the specified rows/columns.
    4.  Return the updated *W_base* for API symmetry.
    """

    if not neuron_indices:
        return W_base

    # Step-1: full-matrix low-rank deltas
    DeltaA_r = _rank_trunc_svd(W_a - W_base, rank)
    DeltaB_r = _rank_trunc_svd(W_b - W_base, rank)
    DeltaM =   1 * DeltaB_r

    # Step-2: selective in-place update
    device = W_base.device
    idx = torch.tensor(list(neuron_indices), dtype=torch.long, device=device)

    if index_dim == 0:
        max_dim = W_base.shape[0]
        idx_valid = [i for i in neuron_indices if (i // div) < max_dim]
        idx = torch.tensor(idx_valid, dtype=torch.long, device=W_base.device)
        W_base[idx // div, :] = (W_base[idx // div, :] + DeltaM[idx // div, :]).to(W_base.dtype)

    else:
        max_dim = W_base.shape[1]
        idx_valid = [i for i in neuron_indices if (i // div) < max_dim]
        idx = torch.tensor(idx_valid, dtype=torch.long, device=W_base.device)
        W_base[:, idx // div] = (W_base[:, idx // div] + DeltaM[:, idx // div]).to(W_base.dtype)

    return W_base


@torch.no_grad()
def merge_with_lora_plus_single(
    W_base: torch.Tensor, W_a: torch.Tensor, W_b: torch.Tensor,
    neuron_indices: Iterable[int],
    index_dim: int,
    tau: float = 0.90,
    beta: float = 0.5,
    svt_lambda_ratio: float = 0.05,
    div: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    LoRA-plus (single branch): fuse W_b with W_base for selected neurons only.
    Returns the same W_base handle (in-place).
    """
    if not neuron_indices:
        return W_base

    device, dtype = W_base.device, W_base.dtype
    neuron_indices = list(neuron_indices)

    # ---------- 1. Low-rank + SVT ----------
    def _svd_trunc_svt(D: torch.Tensor) -> torch.Tensor:
        U, S, Vt = torch.linalg.svd(D.to(torch.float32), full_matrices=False)
        # Select rank by energy threshold tau
        cum = torch.cumsum(S ** 2, dim=0)
        r = int((cum < tau * cum[-1]).sum() + 1)
        r = max(1, r)
        r = min(r, 32)
        # Truncate and soft-threshold
        U, S, Vt = U[:, :r], S[:r], Vt[:r]
        lam = svt_lambda_ratio * S.median()
        S = torch.relu(S - lam)
        return U @ torch.diag(S) @ Vt           # Low-rank denoised delta

    Delta_r = _svd_trunc_svt(W_b - W_base)      # (full matrix)

    # ---------- 2. Selective write-back ----------
    for idx in neuron_indices:
        if index_dim == 0:       # row / out-features
            delta_i = Delta_r[idx // div]
            W_base[idx // div] += (beta * delta_i).to(dtype)
        else:                    # column / in-features
            delta_i = Delta_r[:, idx // div]
            W_base[:, idx // div] += (beta * delta_i).to(dtype)

    return W_base


@torch.no_grad()
def merge_with_lora_stable(
        W_base: torch.Tensor,
        W_a: torch.Tensor,  # Reserved for interface, unused
        W_b: torch.Tensor,
        neuron_indices: Iterable[int],
        index_dim: int,
        energy_ratio: float = 0.95,  # Energy kept after truncation
        beta: float = 1.0,  # Final write-back coefficient
        clip_ratio: float = 0.10,  # Allow weight norm +/- 10%
        max_rank: int = 32,
        div: int = 1,  # When some neurons are oversampled
        dtype=torch.float32,
):
    """
    Full low-rank SVD, local write-back.
    """
    if not neuron_indices:
        return W_base

    device = W_base.device
    dtype0 = W_base.dtype
    neuron_indices = list(neuron_indices)

    # ---------- 1. Full Delta + SVD truncation ----------
    def _svd_energy_trunc(D: torch.Tensor) -> torch.Tensor:
        U, S, Vt = torch.linalg.svd(D.to(dtype), full_matrices=False)
        cum = torch.cumsum(S ** 2, dim=0)
        total_energy = cum[-1]
        r = int((cum < energy_ratio * total_energy).sum() + 1)
        r = min(max(1, r), max_rank)
        return (U[:, :r] @ torch.diag(S[:r]) @ Vt[:r, :]).to(dtype0)

    Delta_r = _svd_energy_trunc(W_b - W_base)

    # ---------- 2. Local write-back ----------
    for idx in neuron_indices:
        if index_dim == 0:  # row / out-features
            delta_i = Delta_r[idx // div]
            W_base[idx // div] += (beta * delta_i).to(dtype0)
        else:  # column / in-features
            delta_i = Delta_r[:, idx // div]
            W_base[:, idx // div] += (beta * delta_i).to(dtype0)

    # ---------- 3. Norm clipping ----------
    return W_base



@torch.no_grad()
def merge_with_lora_plus_AB(
    W_base: torch.Tensor, W_a: torch.Tensor, W_b: torch.Tensor,
    neuron_indices: Iterable[int], index_dim: int,
    tau: float = 0.90,   # Energy threshold
    beta_global: float = 0.5,
    svt_lambda_ratio: float = 0.05,  # λ = ratio * median(σ)
    div: int = 1, eps: float = 1e-6
):
    if not neuron_indices:
        return W_base
    device, dtype = W_base.device, W_base.dtype

    def _svd_trunc_svt(D: torch.Tensor):
        U, S, Vt = torch.linalg.svd(D.to(torch.float32), full_matrices=False)
        # energy-based r
        energy = torch.cumsum(S**2, 0)
        r = int((energy < tau * energy[-1]).sum() + 1)
        r = max(1, r)
        U, S, Vt = U[:, :r], S[:r], Vt[:r]
        # soft-threshold
        lam = svt_lambda_ratio * S.median()
        S = torch.relu(S - lam)
        return U @ torch.diag(S) @ Vt

    ΔA_r = _svd_trunc_svt(W_a - W_base)
    ΔB_r = _svd_trunc_svt(W_b - W_base)

    # Gram-Schmidt orthogonalization (approximate)
    proj = (ΔA_r * ΔB_r).sum() / (ΔA_r.norm()**2 + eps)
    ΔB_ortho = ΔB_r - proj * ΔA_r

    for i in neuron_indices:
        if index_dim == 0:
            a_i = ΔA_r[i // div]
            b_i = ΔB_ortho[i // div]
            beta_i = (a_i.norm()**2) / (a_i.norm()**2 + b_i.norm()**2 + eps)
            delta = (beta_i * beta_global) * a_i + (1 - beta_i * beta_global) * b_i
            W_base[i // div] += delta.to(dtype)
        else:
            a_i = ΔA_r[:, i // div]
            b_i = ΔB_ortho[:, i // div]
            beta_i = (a_i.norm()**2) / (a_i.norm()**2 + b_i.norm()**2 + eps)
            delta = (beta_i * beta_global) * a_i + (1 - beta_i * beta_global) * b_i
            W_base[:, i // div] += delta.to(dtype)

    return W_base


REDUCTION_FN: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
    lambda a, b: (0 * a + 1 * b) * 1
)

def fetch_param(sd, name: str):
    """Fetch weight from a possibly different hierarchy.

    Handles common HF architectural prefix differences between LLaMA-family,
    Idefics3 (text tower), and Qwen-style models by trying a few canonical
    rewrites before giving up.
    """
    if name in sd:
        return sd[name]

    candidates = [name]
    # Common language-model shims
    repls = [
        ("model.language_model.layers.", "model.layers."),
        ("language_model.model.layers.", "model.layers."),
        ("language_model.layers.", "model.layers."),
        ("model.text_model.layers.", "model.layers."),      # idefics3 text tower
        ("text_model.model.layers.", "model.layers."),
        ("model.decoder.layers.", "model.layers."),
        ("llm.model.layers.", "model.layers.")
    ]
    for a, b in repls:
        candidates.append(name.replace(a, b, 1))

    for key in candidates:
        if key in sd:
            return sd[key]
    return None


###############################################################################
# Core merging                                                                #
###############################################################################

def merge_shared_neurons(
    base_model: torch.nn.Module,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    neurons_shared: Dict[str, Dict[str, Set[int]]],
    merge_method: str = "mean",
    pca_k: int = 10,
    pca_alpha: float = 0.5,
    lora_rank: int = 4,
    lora_beta: float = 0.5,
) -> None:
    """In-place merge of *shared* neurons from A & B into *base_model*."""

    real_base = base_model.module if hasattr(base_model, "module") else base_model
    cfg: AutoConfig = real_base.config
    print(cfg)

    # Robust kv_factor inference.
    num_heads = getattr(cfg, "num_attention_heads", None)
    num_kv = getattr(cfg, "num_key_value_heads", None)
    print(num_heads)
    print(num_kv)
    # if isinstance(num_heads, int) and isinstance(num_kv, int) and num_kv > 0:
    #     kv_factor = 4
    # else:
    #     kv_factor = 1
    kv_factor = 4
    # Mapping from weight name patterns to JSON keys and division factors.
    proj_map = {
        "attn.q_proj": ("attn_q", 1),
        "attn.k_proj": ("attn_k", kv_factor),
        "attn.v_proj": ("attn_v", kv_factor),
        "attn.o_proj": ("attn_o", 1),
        "up_proj":     ("fwd_up", 1),
        "down_proj":   ("fwd_down", 1),

        # ========= Uncomment aliases below if you need to support more prefixes =========
        # "self_attn.q_proj": ("attn_q", 1),
        # "self_attn.k_proj": ("attn_k", kv_factor),
        # "self_attn.v_proj": ("attn_v", kv_factor),
        # "self_attn.o_proj": ("attn_o", 1),
        # "mlp.gate_proj": ("fwd_up", 1),
        # "mlp.down_proj": ("fwd_down", 1),
    }

    sd_base = base_model.state_dict()
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    # Debug prints for parameter inspection; comment out if not needed.
    print("Base Model Parameter Names:")
    for name in sd_base.keys():
        print(name)

    print("\nModel A Parameter Names:")
    for name in sd_a.keys():
        print(name)

    print("\nModel B Parameter Names:")
    for name in sd_b.keys():
        print(name)

    with torch.no_grad():
        for full_name, W_base in sd_base.items():
            W_a = fetch_param(sd_a, full_name)
            W_b = fetch_param(sd_b, full_name)

            if W_a is None or W_b is None:
                print(f"Skip {full_name} (not found in A or B)")
                continue

            if W_base.dim() == 1:  # bias
                continue

            m = re.search(r"layers\.(\d+)\.", full_name)
            if not m:
                continue
            layer = m.group(1)
            print(f"🔧 Prepare Merging {full_name}  [method={merge_method}]")
            for sub, (json_key, div) in proj_map.items():
                # print(sub,full_name)
                if sub not in full_name:
                    continue

                neurons_layer = neurons_shared.get(json_key, {}).get(layer, set())
                # print(neurons_shared,json_key)
                # print(neurons_layer)
                if not neurons_layer:
                    break
                print(f"🔧 Merging {full_name}  [method={merge_method}]")
                # down_proj & attn.o_proj: column update; others: row update
                index_dim = 1 if ("down_proj" in sub or "attn.o_proj" in sub) else 0

                print('full_name', full_name)

                if merge_method == "mean":
                    for n in neurons_layer:
                        dest_idx = int(n // div)
                        if index_dim == 0:
                            slice_a = W_a[dest_idx, :]
                            slice_b = W_b[dest_idx, :]
                            W_base[dest_idx, :] = REDUCTION_FN(slice_a, slice_b).to(dtype=W_base.dtype)
                        else:
                            slice_a = W_a[:, dest_idx]
                            slice_b = W_b[:, dest_idx]
                            W_base[:, dest_idx] = REDUCTION_FN(slice_a, slice_b).to(dtype=W_base.dtype)

                elif merge_method == "lora":
                    print('MERGE_LORA')
                    W_merged = merge_with_lora(
                        W_base,
                        W_a,
                        W_b,
                        neurons_layer,
                        index_dim=index_dim,
                        rank=lora_rank,
                        beta=lora_beta,
                        div=div
                    )
                    W_base.copy_(W_merged)

                break  # Exit after matching one substring, move to next param

###############################################################################
# CLI                                                                         #
###############################################################################
#
# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(
#         description="Merge shared neurons from two models into a base model."
#     )
#     p.add_argument("--base", default="openbmb/MiniCPM-Llama3-V-2_5", help="Path to base model")
#     p.add_argument("--model_a", default="MergeBench/Llama-3.1-8B_math", help="Path to model A")
#     p.add_argument("--model_b", default="openbmb/MiniCPM-Llama3-V-2_5", help="Path to model B")
#     p.add_argument("--json",  default="/mnt/data/jingnan/cch/detect_neruon/neuron_train_data_detect/compare_Llama-3.1-8B_math_gsm_en_atten0.1_ffn0.1_vs_MiniCPM-8B-Llama3_multimodal_en_atten0.1_ffn0.1/shared_VL_reasoning.json", help="JSON file listing shared neurons")
#     p.add_argument("--output_dir", default="shared_merged_model_minicpm_lora_merge", help="Directory to save merged model")
#
#     p.add_argument("--merge_method", default="lora",
#                    choices=["mean", "pca", "lora", "lora_plus_single", "lora_plus_multi", "lora_stable"],
#                    help="Fusion strategy")
#
#     p.add_argument("--pca_k", type=int, default=10, help="# principal components (pca mode)")
#     p.add_argument("--pca_alpha", type=float, default=0.5, help="Interpolation α (pca mode)")
#
#     p.add_argument("--lora_rank", type=int, default=4, help="Rank r for LoRA (lora mode)")
#     p.add_argument("--lora_beta", type=float, default=0.5, help="Interpolation β (lora mode)")
#
#     # misc
#     p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Precision")
#     p.add_argument("--device", default="cuda:0", help="Device map for model loading")
#     return p.parse_args()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge shared neurons from two models into a base model."
    )
    p.add_argument(
        "--base",
        default="llava-hf/llama3-llava-next-8b-hf",
        help="Path to base model",
    )
    p.add_argument(
        "--model_a",
        default="llava-hf/llama3-llava-next-8b-hf",
        help="Path to model A",
    )
    p.add_argument(
        "--model_b",
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Path to model B",
    )
    p.add_argument(
        "--json",
        default="./shared_VL_reasoning.json",
        help="JSON file listing shared neurons",
    )
    p.add_argument(
        "--output_dir",
        default="shared_merged_model_llavanext_lora_merge",
        help="Directory to save merged model",
    )

    p.add_argument("--merge_method", default="lora",
                   choices=["mean", "lora"],
                   help="Fusion strategy")

    p.add_argument("--pca_k", type=int, default=10, help="# principal components (pca mode)")
    p.add_argument("--pca_alpha", type=float, default=0.5, help="Interpolation α (pca mode)")

    p.add_argument("--lora_rank", type=int, default=8, help="Rank r for LoRA (lora mode)")
    p.add_argument("--lora_beta", type=float, default=1, help="Interpolation β (lora mode)")

    # misc
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Precision")
    p.add_argument("--device", default="auto", help="Device map for model loading")
    return p.parse_args()
###############################################################################
# Main                                                                         #
###############################################################################

def main() -> None:
    args = parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print("Loading models …", flush=True)

    # Smart loading (Qwen2.5-VL / InternVL2 / InternVL2.5 / others)
    model_base = smart_load_model(args.base, dtype, args.device)
    model_a    = smart_load_model(args.model_a, dtype, args.device)
    model_b    = smart_load_model(args.model_b, dtype, args.device)

    neurons_shared = read_neuron(args.json)
    print(f"Merging shared neurons … (method={args.merge_method})", flush=True)

    merge_shared_neurons(
        model_base,
        model_a,
        model_b,
        neurons_shared,
        merge_method=args.merge_method,
        pca_k=args.pca_k,
        pca_alpha=args.pca_alpha,
        lora_rank=args.lora_rank,
        lora_beta=args.lora_beta,
    )

    print(f"Saving merged model to {args.output_dir} …", flush=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model_base.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()