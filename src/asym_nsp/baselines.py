"""Symmetric baseline LoRA merging methods.

Implements faithful reproductions of:
  - Naive Addition (sum of deltas)
  - Task Arithmetic  (Ilharco et al., ICLR 2023)
  - TIES-Merging    (Yadav et al., NeurIPS 2023, Algorithm 1)
  - DARE            (Yu et al., ICML 2024, Eq. 1 & 3)

All functions operate on delta-weight tensors (ΔW = B @ A) or on lists of such
tensors. The *lora_adapters* variant works directly on LoRAAdapter objects.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import torch

from asym_nsp.lora_loader import LoRAAdapter


# ══════════════════════════════════════════════════════════════════════════════
# Primitive merge functions  (operate on List[torch.Tensor] of delta weights)
# ══════════════════════════════════════════════════════════════════════════════

def merge_naive_addition(deltas: List[torch.Tensor]) -> torch.Tensor:
    """Element-wise sum of all delta-weight tensors.

    Args:
        deltas: List of tensors with identical shape.  deltas[0] is the Master
                ΔW_m; deltas[1:] are Plugin ΔW_{p_i}.

    Returns:
        Merged delta tensor: Σ_t ΔW_t.
    """
    if not deltas:
        raise ValueError("deltas must be non-empty")
    result = deltas[0].clone()
    for d in deltas[1:]:
        result = result + d
    return result


def merge_task_arithmetic(
    deltas: List[torch.Tensor],
    scaling_lam: float = 0.5,
) -> torch.Tensor:
    """Task Arithmetic (Ilharco et al., ICLR 2023): λ · Σ_t ΔW_t.

    Args:
        deltas: List of delta-weight tensors.
        scaling_lam: Global scaling coefficient λ.

    Returns:
        Merged delta tensor scaled by λ.
    """
    if not deltas:
        raise ValueError("deltas must be non-empty")
    return scaling_lam * merge_naive_addition(deltas)


def merge_ties(
    deltas: List[torch.Tensor],
    top_k_pct: float = 20.0,
    scaling_lam: float = 1.0,
) -> torch.Tensor:
    """TIES-Merging (Yadav et al., NeurIPS 2023, Algorithm 1).

    Three-step procedure applied symmetrically to all deltas:
      1. Trim  – keep only the top `top_k_pct`% elements by |magnitude|.
      2. Elect – compute aggregate sign γ = sgn(Σ_t τ̂_t).
      3. Disjoint Merge – average only the τ̂_t with sign == γ (ignoring zeros),
                          then scale by `scaling_lam`.

    NOTE: We deliberately avoid torch.quantile() because it raises a RuntimeError
    for tensors with more than 16M elements (a known PyTorch limitation on CUDA).
    Instead we use torch.topk() to find the magnitude threshold.

    Args:
        deltas:      List of delta-weight tensors (same shape).
        top_k_pct:   Percentage of parameters to keep per delta (0 < pct ≤ 100).
        scaling_lam: Final scaling factor λ.

    Returns:
        Merged delta tensor.
    """
    if not deltas:
        raise ValueError("deltas must be non-empty")
    if not (0.0 < top_k_pct <= 100.0):
        raise ValueError(f"top_k_pct must be in (0, 100], got {top_k_pct}")

    shape = deltas[0].shape
    dtype = deltas[0].dtype
    device = deltas[0].device
    n_total = deltas[0].numel()

    # ── Step 1: Trim ──────────────────────────────────────────────────────────
    trimmed: List[torch.Tensor] = []
    for delta in deltas:
        flat_abs = delta.abs().view(-1).float()
        k = max(1, int(round(n_total * top_k_pct / 100.0)))
        k = min(k, n_total)
        # Find the k-th largest absolute value as threshold
        # topk is O(n) in PyTorch and works for arbitrary tensor sizes
        threshold_val = torch.topk(flat_abs, k, largest=True, sorted=False).values.min()
        mask = (delta.abs() >= threshold_val).to(dtype=delta.dtype)
        trimmed.append(delta * mask)

    # ── Step 2: Elect ─────────────────────────────────────────────────────────
    stacked = torch.stack(trimmed, dim=0)          # [T, *shape]
    aggregate = stacked.sum(dim=0)                 # [*shape]
    gamma = torch.sign(aggregate)                  # {-1, 0, +1}

    # ── Step 3: Disjoint Merge ────────────────────────────────────────────────
    # Keep τ̂_t only where sign(τ̂_t) == γ  (zeros and wrong-sign values → 0)
    gamma_expanded = gamma.unsqueeze(0).expand_as(stacked)
    sign_match = (torch.sign(stacked) == gamma_expanded) & (stacked != 0)
    masked = stacked * sign_match.to(dtype=stacked.dtype)

    count = sign_match.float().sum(dim=0)          # [*shape], count of agreeing entries
    # Avoid division by zero: positions with count==0 stay 0
    merged = torch.where(count > 0, masked.sum(dim=0) / count, torch.zeros_like(aggregate))
    return (scaling_lam * merged).to(dtype=dtype, device=device)


def merge_dare(
    deltas: List[torch.Tensor],
    drop_rate: float = 0.5,
    base_method: str = "task_arithmetic",
    scaling_lam: float = 0.5,
    top_k_pct: float = 20.0,
    seed: int = 42,
) -> torch.Tensor:
    """DARE (Yu et al., ICML 2024, Eq. 1 & 3).

    For each delta independently:
      1. Randomly drop elements with probability `drop_rate`.
      2. Rescale the surviving values by 1 / (1 - drop_rate) to keep the
         expectation unbiased.
    Then merge the resulting sparse deltas using `base_method`.

    Args:
        deltas:      List of delta-weight tensors.
        drop_rate:   Probability of dropping each parameter (0 ≤ p < 1).
        base_method: Downstream merge: "task_arithmetic" | "average" | "ties".
        scaling_lam: Scaling coefficient forwarded to the base method.
        top_k_pct:   Top-k percentage forwarded to TIES when base_method="ties".
        seed:        Base random seed (each delta gets seed + idx for independence).

    Returns:
        Merged delta tensor.
    """
    if not (0.0 <= drop_rate < 1.0):
        raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
    if base_method not in {"task_arithmetic", "average", "ties"}:
        raise ValueError(f"base_method must be 'task_arithmetic', 'average', or 'ties', got {base_method}")
    if not deltas:
        raise ValueError("deltas must be non-empty")

    rescale = 1.0 / (1.0 - drop_rate)
    sparse_deltas: List[torch.Tensor] = []
    for idx, delta in enumerate(deltas):
        gen = torch.Generator(device=delta.device)
        gen.manual_seed(seed + idx)
        keep_mask = torch.bernoulli(
            torch.full(delta.shape, 1.0 - drop_rate, dtype=torch.float32, device=delta.device),
            generator=gen,
        ).to(dtype=delta.dtype)
        sparse_deltas.append(delta * keep_mask * rescale)

    if base_method == "task_arithmetic":
        return merge_task_arithmetic(sparse_deltas, scaling_lam=scaling_lam)
    elif base_method == "average":
        total = torch.zeros_like(sparse_deltas[0])
        for d in sparse_deltas:
            total = total + d
        return total / len(sparse_deltas)
    else:  # "ties"
        return merge_ties(sparse_deltas, top_k_pct=top_k_pct, scaling_lam=scaling_lam)


# ══════════════════════════════════════════════════════════════════════════════
# High-level adapter-level baseline merge
# ══════════════════════════════════════════════════════════════════════════════

def merge_lora_adapters_baseline(
    master: LoRAAdapter,
    plugins: List[LoRAAdapter],
    method: str,
    target_modules: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Merge LoRAAdapter objects using a symmetric baseline method.

    For each module that appears in `master` (and optionally `target_modules`),
    gathers ΔW = scaling * (B @ A) from master and each plugin adapter, then
    merges using the specified method.

    Args:
        master:         Master LoRAAdapter.
        plugins:        List of plugin LoRAAdapters.
        method:         One of "naive", "task_arithmetic", "ties", "dare".
        target_modules: If provided, restrict to this subset of module keys.
        **kwargs:       Extra keyword arguments forwarded to the merge function
                        (e.g. scaling_lam, top_k_pct, drop_rate, seed).

    Returns:
        Dict mapping module_key -> merged ΔW tensor [d_out, d_in].
    """
    valid_methods = {"naive", "task_arithmetic", "ties", "dare"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}")

    module_keys = list(master.layers.keys())
    if target_modules is not None:
        module_keys = [k for k in module_keys if k in target_modules]

    merged: Dict[str, torch.Tensor] = {}

    for mod_key in module_keys:
        master_layer = master.get_layer(mod_key)
        if master_layer is None:
            continue

        # Collect delta-weight tensors: [ΔW_master, ΔW_p1, ΔW_p2, ...]
        deltas: List[torch.Tensor] = [master_layer.compute_delta_w()]
        for plugin in plugins:
            plugin_layer = plugin.get_layer(mod_key)
            if plugin_layer is not None:
                deltas.append(plugin_layer.compute_delta_w())

        if method == "naive":
            merged[mod_key] = merge_naive_addition(deltas)
        elif method == "task_arithmetic":
            merged[mod_key] = merge_task_arithmetic(deltas, **kwargs)
        elif method == "ties":
            merged[mod_key] = merge_ties(deltas, **kwargs)
        elif method == "dare":
            merged[mod_key] = merge_dare(deltas, **kwargs)

    return merged


__all__ = [
    "merge_naive_addition",
    "merge_task_arithmetic",
    "merge_ties",
    "merge_dare",
    "merge_lora_adapters_baseline",
]

