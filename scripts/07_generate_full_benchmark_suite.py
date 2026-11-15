"""scripts/07_generate_full_benchmark_suite.py
===========================================================
Generates 4 publication-ready figures (300 DPI) for the Asymmetric NSP paper.

Figures produced in assets/
  1. baseline_comparison.png          – Asym-NSP vs Naive / Task-Arithmetic / TIES / DARE
  2. rank_ablation_pareto.png         – Pareto front: Master protection vs Plugin energy over k
  3. layerwise_interference_profile.png – per-layer interference across all 32 Transformer layers
  4. multi_adapter_capacity.png       – null-space capacity when cascading N=1..8 plugin adapters

ALL numbers come from real measurements — no hardcoded timings, no sine waves.
CPU-only. Runtime: ~5–60 s depending on machine (Plot 3 is the heaviest).
===========================================================
"""

import os
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Make the src package importable ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asym_nsp.baselines import (
    merge_dare,
    merge_naive_addition,
    merge_task_arithmetic,
    merge_ties,
)
from asym_nsp.core import (
    build_cumulative_subspace_basis,
    compute_subspace_basis,
    project_two_sided_nsp,
)

# ── Global style ──────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "figure.dpi":       300,
})

SAVE_DIR = Path(__file__).resolve().parent.parent / "assets"
SAVE_DIR.mkdir(exist_ok=True)

# ── Colour palette (colour-blind-friendly) ────────────────────────────────────
C_NSP   = "#2166AC"   # our method  – blue
C_NAIVE = "#D6604D"   # naive       – red-orange
C_TA    = "#F4A582"   # Task Arith  – light orange
C_TIES  = "#92C5DE"   # TIES        – light blue
C_DARE  = "#4DAC26"   # DARE        – green

torch.manual_seed(42)
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_loras(d=4096, r_m=16, r_p=16, interference=0.5, seed=42):
    """Synthetic LoRA matrices with controlled interference."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    B_m = torch.randn(d, r_m, generator=gen)
    A_m = torch.randn(r_m, d, generator=gen)
    B_p = torch.randn(d, r_p, generator=gen) + interference * B_m[:, :r_p]
    A_p = torch.randn(r_p, d, generator=gen) + interference * A_m[:r_p, :]
    return B_m, A_m, B_p, A_p


def _master_output_error(B_m, A_m, B_p, A_p, n_tests=64):
    """Relative error of merged model on Master-aligned inputs."""
    r_m = A_m.shape[0]
    _, _, Vh = torch.linalg.svd(A_m.float(), full_matrices=False)
    V_m = Vh  # [r_m, d_model]
    gen = torch.Generator(); gen.manual_seed(0)
    x = torch.matmul(torch.randn(n_tests, r_m, generator=gen), V_m)

    dW_m = B_m.float() @ A_m.float()
    dW_p = B_p.float() @ A_p.float()
    y_ref   = x @ dW_m.T
    y_merge = x @ (dW_m + dW_p).T
    return (torch.norm(y_merge - y_ref) / torch.norm(y_ref)).item()


def _plugin_energy_sq_pct(dW_p_raw, dW_p_out):
    """Squared-Frobenius plugin energy retained (Eq. 7, in %)."""
    num = float(torch.sum(dW_p_out.float() ** 2))
    den = float(torch.sum(dW_p_raw.float() ** 2))
    return num / (den + 1e-30) * 100.0


def _time_fn(fn, n_warmup=1, n_runs=5):
    """Wall-clock timing: 1 warmup + mean of n_runs (ms)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASELINE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_baselines():
    """Measure real metrics and real CPU latency for all 5 methods."""
    d, r_m, r_p = 4096, 16, 16
    B_m, A_m, B_p_raw, A_p_raw = _make_loras(d, r_m, r_p, interference=0.6, seed=42)
    dW_m   = B_m.float() @ A_m.float()
    dW_p   = B_p_raw.float() @ A_p_raw.float()
    deltas = [dW_m, dW_p]

    results = {}

    # ── Asym-NSP (ours): time the factor-space projection ─────────────────────
    B_p_nsp, A_p_nsp = project_two_sided_nsp(B_p_raw, A_p_raw, B_m, A_m)
    lat_nsp = _time_fn(lambda: project_two_sided_nsp(B_p_raw, A_p_raw, B_m, A_m))
    dW_p_nsp = B_p_nsp.float() @ A_p_nsp.float()
    results["Asym-NSP\n(Ours)"] = dict(
        master_err    = _master_output_error(B_m, A_m, B_p_nsp, A_p_nsp) * 100,
        plugin_energy = _plugin_energy_sq_pct(dW_p, dW_p_nsp),
        latency       = lat_nsp,
    )

    # ── Naive Addition ────────────────────────────────────────────────────────
    # Delta-space: sum B_p@A_p in delta space (B_p raw → delta, then add)
    lat_naive = _time_fn(lambda: merge_naive_addition(deltas))
    dW_naive  = merge_naive_addition(deltas)
    dW_naive_plugin = dW_naive - dW_m          # = dW_p exactly
    results["Naive\nAddition"] = dict(
        master_err    = _master_output_error(B_m, A_m, B_p_raw, A_p_raw) * 100,
        plugin_energy = _plugin_energy_sq_pct(dW_p, dW_naive_plugin),
        latency       = lat_naive,
    )

    # ── Task Arithmetic (λ=0.5) ───────────────────────────────────────────────
    lat_ta  = _time_fn(lambda: merge_task_arithmetic(deltas, scaling_lam=0.5))
    dW_ta   = merge_task_arithmetic(deltas, scaling_lam=0.5)
    # plugin component from merged result vs master
    dW_ta_plugin = dW_ta - dW_m
    results["Task\nArithmetic"] = dict(
        master_err    = _master_output_error(B_m, A_m, B_p_raw, A_p_raw) * 100 * 0.5,
        plugin_energy = _plugin_energy_sq_pct(dW_p, dW_ta_plugin),
        latency       = lat_ta,
    )
    # Note: TA scales entire sum by λ, so master contribution also shrinks.
    # We re-measure properly: merged = λ(dW_m + dW_p), plugin component = dW_ta - dW_m
    # master_err: input is test_x aligned to V_A(A_m); y_merge = x @ (dW_m + dW_p).T * lam
    #   rel_error = ||y_merge - y_master||/||y_master|| where y_master = x @ dW_m.T
    # This is already captured in _master_output_error with the adjusted plugin:
    # Treat as: effective plugin = dW_ta - dW_m (what gets added to dW_m)
    # Recompute properly:
    r_m_local = A_m.shape[0]
    _, _, Vh_local = torch.linalg.svd(A_m.float(), full_matrices=False)
    gen_local = torch.Generator(); gen_local.manual_seed(0)
    x_local = torch.randn(64, r_m_local, generator=gen_local) @ Vh_local
    y_master_local  = x_local @ dW_m.T
    y_merge_ta      = x_local @ dW_ta.T
    err_ta = (torch.norm(y_merge_ta - y_master_local) / torch.norm(y_master_local)).item() * 100
    results["Task\nArithmetic"]["master_err"] = err_ta

    # ── TIES-Merging (k=20%, λ=1.0) ───────────────────────────────────────────
    lat_ties  = _time_fn(lambda: merge_ties(deltas, top_k_pct=20.0, scaling_lam=1.0))
    dW_ties   = merge_ties(deltas, top_k_pct=20.0, scaling_lam=1.0)
    dW_ties_plugin = dW_ties - dW_m
    y_merge_ties = x_local @ dW_ties.T
    err_ties = (torch.norm(y_merge_ties - y_master_local) / torch.norm(y_master_local)).item() * 100
    results["TIES\nMerging"] = dict(
        master_err    = err_ties,
        plugin_energy = _plugin_energy_sq_pct(dW_p, dW_ties_plugin),
        latency       = lat_ties,
    )

    # ── DARE (p=0.5, TA λ=0.5) ────────────────────────────────────────────────
    lat_dare  = _time_fn(lambda: merge_dare(deltas, drop_rate=0.5,
                                             base_method="task_arithmetic",
                                             scaling_lam=0.5, seed=42))
    dW_dare   = merge_dare(deltas, drop_rate=0.5, base_method="task_arithmetic",
                            scaling_lam=0.5, seed=42)
    dW_dare_plugin = dW_dare - dW_m
    y_merge_dare = x_local @ dW_dare.T
    err_dare = (torch.norm(y_merge_dare - y_master_local) / torch.norm(y_master_local)).item() * 100
    results["DARE"] = dict(
        master_err    = err_dare,
        plugin_energy = _plugin_energy_sq_pct(dW_p, dW_dare_plugin),
        latency       = lat_dare,
    )

    return results


def plot_baseline_comparison():
    print("  [1/4] Generating baseline_comparison.png ...")
    results = _simulate_baselines()

    # Print numeric summary
    print("\n  ── Baseline Comparison Summary ──────────────────────────────")
    print(f"  {'Method':<22}  {'Master Shift (%)':>16}  {'Plugin Energy (%)':>18}  {'Latency (ms)':>14}")
    print(f"  {'-'*22}  {'-'*16}  {'-'*18}  {'-'*14}")
    for mname, m in results.items():
        label = mname.replace('\n', ' ')
        print(f"  {label:<22}  {m['master_err']:>16.4f}  {m['plugin_energy']:>18.2f}  {m['latency']:>14.3f}")
    print()

    methods = list(results.keys())
    master_pres   = [100.0 - m["master_err"]    for m in results.values()]
    plugin_energy = [m["plugin_energy"]           for m in results.values()]
    latencies     = [m["latency"]                 for m in results.values()]

    # Map method name → colour
    color_map = {
        "Asym-NSP\n(Ours)":   C_NSP,
        "Naive\nAddition":     C_NAIVE,
        "Task\nArithmetic":    C_TA,
        "TIES\nMerging":       C_TIES,
        "DARE":                C_DARE,
    }
    colors = [color_map.get(m, "#888888") for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle(
        "Baseline Comparison: Asymmetric NSP vs. Competing Methods\n"
        "(all numbers from real measurements on 4096×4096 synthetic LoRA layer)",
        fontsize=12, fontweight="bold", y=1.03,
    )

    metric_triples = [
        ("Master Preservation\n(%, higher is better)", master_pres),
        ("Plugin Energy Retained\n(%, higher is better)\n"
         "[= ||ΔW_merged − ΔW_m||²_F / ||ΔW_p||²_F × 100]", plugin_energy),
        ("Merge Latency\n(ms per layer, lower is better)", latencies),
    ]

    for ax, (metric_label, vals) in zip(axes, metric_triples):
        bars = ax.bar(
            range(len(methods)), vals,
            color=colors, width=0.6, zorder=3,
            edgecolor="white", linewidth=0.8,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )
        bars[0].set_edgecolor("#001F5B")
        bars[0].set_linewidth(2.0)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=8, fontweight="bold")
        ax.set_ylabel(metric_label.split("\n")[0], fontsize=9)
        ax.set_title(metric_label, fontsize=8, pad=8)

    plt.tight_layout()
    path = SAVE_DIR / "baseline_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"     -> Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. RANK ABLATION PARETO FRONT
# ══════════════════════════════════════════════════════════════════════════════

def plot_rank_ablation_pareto():
    print("  [2/4] Generating rank_ablation_pareto.png ...")

    d, r_m = 4096, 16
    B_m, A_m, B_p_raw, A_p_raw = _make_loras(d, r_m, r_m, interference=0.6, seed=42)
    dW_p_raw = B_p_raw.float() @ A_p_raw.float()

    k_vals        = list(range(1, r_m + 1))
    master_prot   = []
    plugin_energy = []

    print("\n  ── Rank Ablation Summary ────────────────────────────────────")
    print(f"  {'k':>4}  {'Master Prot (%)':>16}  {'Plugin Energy (%)':>18}")
    print(f"  {'-'*4}  {'-'*16}  {'-'*18}")

    for k in k_vals:
        # Left projection
        U_k, _ = compute_subspace_basis(B_m, rank=k, basis_side="left")
        B_p_proj = B_p_raw.float() - U_k.float() @ (U_k.float().T @ B_p_raw.float())
        # Right projection
        V_k, _ = compute_subspace_basis(A_m, rank=k, basis_side="right")
        A_p_proj = A_p_raw.float() - (A_p_raw.float() @ V_k.float()) @ V_k.float().T

        err  = _master_output_error(B_m, A_m, B_p_proj, A_p_proj) * 100
        dW_p_proj = B_p_proj @ A_p_proj
        eng  = _plugin_energy_sq_pct(dW_p_raw, dW_p_proj)
        master_prot.append(100 - err)
        plugin_energy.append(eng)
        print(f"  {k:>4}  {100-err:>16.4f}  {eng:>18.4f}")

    # Report k=1 and k=16 explicitly
    print(f"\n  k=1  → Master Prot: {master_prot[0]:.4f}%,  Plugin Energy: {plugin_energy[0]:.4f}%")
    print(f"  k={r_m} → Master Prot: {master_prot[-1]:.4f}%,  Plugin Energy: {plugin_energy[-1]:.4f}%\n")

    master_prot   = np.array(master_prot)
    plugin_energy = np.array(plugin_energy)
    k_vals_arr    = np.array(k_vals)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Rank Ablation: Master Protection vs. Plugin Energy (Pareto Front)\n"
        f"(d={d}, r_m={r_m}; Plugin Energy = ||ΔW_p'||²_F / ||ΔW_p||²_F × 100)",
        fontsize=12, fontweight="bold", y=1.03,
    )

    sc = ax_left.scatter(
        plugin_energy, master_prot,
        c=k_vals_arr, cmap="plasma", s=80, zorder=5,
        edgecolors="white", linewidths=0.6,
    )
    cbar = fig.colorbar(sc, ax=ax_left, pad=0.02)
    cbar.set_label("Protected rank k", fontsize=9)

    ax_left.annotate(
        f"k = r_m = {r_m}\n(Asym-NSP default)",
        xy=(plugin_energy[-1], master_prot[-1]),
        xytext=(plugin_energy[-1] - 8, master_prot[-1] - 4),
        fontsize=9, color=C_NSP, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_NSP, lw=1.2),
    )
    ax_left.scatter([plugin_energy[-1]], [master_prot[-1]],
                    s=160, c=C_NSP, zorder=6, marker="*")
    ax_left.set_xlabel("Plugin Energy Retained (%) = ||ΔW_p'||²/||ΔW_p||² × 100", fontsize=9)
    ax_left.set_ylabel("Master Preservation (%)", fontsize=10)
    ax_left.set_title("Pareto Front  (upper-right = ideal)", fontsize=10)

    ax2 = ax_right.twinx()
    ax_right.plot(k_vals_arr, master_prot, color=C_NSP, marker="o", lw=2,
                  label="Master Preservation (%)")
    ax2.plot(k_vals_arr, plugin_energy, color="#B2182B", marker="s", lw=2,
             linestyle="--", label="Plugin Energy (%)")

    ax_right.set_xlabel("Protected rank k", fontsize=10)
    ax_right.set_ylabel("Master Preservation (%)", color=C_NSP, fontsize=10)
    ax2.set_ylabel("Plugin Energy Retained (%) = ||ΔW_p'||²/||ΔW_p||² × 100",
                   color="#B2182B", fontsize=9)
    ax_right.tick_params(axis="y", labelcolor=C_NSP)
    ax2.tick_params(axis="y", labelcolor="#B2182B")
    ax_right.axvline(r_m, color="gray", linestyle=":", alpha=0.7)
    ax_right.text(r_m + 0.2, master_prot.min() + 1, f"k = {r_m}", fontsize=9, color="gray")

    lines1, labels1 = ax_right.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_right.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
    ax_right.set_title("Protection vs Energy Trade-off", fontsize=10)

    plt.tight_layout()
    path = SAVE_DIR / "rank_ablation_pareto.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"     -> Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. LAYER-WISE INTERFERENCE PROFILE  (32-layer LLaMA-3-8B dimensions)
# ══════════════════════════════════════════════════════════════════════════════

def plot_layerwise_interference():
    print("  [3/4] Generating layerwise_interference_profile.png ...")
    print("        (Simulating true LLaMA-3-8B dims, running project_two_sided_nsp per sub-layer)")

    n_layers = 32
    d_model  = 4096
    d_mlp    = 14336
    r_m      = 16
    r_p      = 16

    # Sub-layer shapes:
    #   Attn: q/k/v/o  → [d_model, d_model]
    #   MLP:  gate/up  → [d_mlp,  d_model]
    #         down     → [d_model, d_mlp]
    attn_shape = (d_model, d_model)   # 4 sub-layers per transformer layer
    mlp_shapes = [
        (d_mlp,  d_model),  # gate_proj
        (d_mlp,  d_model),  # up_proj
        (d_model, d_mlp),   # down_proj
    ]

    def _relative_left_leakage(B_m, B_p):
        """||U_m^T B_p||_F / ||B_p||_F × 100  (before projection)."""
        U_m, _ = compute_subspace_basis(B_m, basis_side="left")
        if U_m.shape[1] == 0:
            return 0.0
        num = float(torch.norm(U_m.float().T @ B_p.float(), p="fro"))
        den = float(torch.norm(B_p.float(), p="fro"))
        return num / (den + 1e-12) * 100.0

    def _relative_left_leakage_after(U_m, B_p_proj):
        """||U_m^T B_p'||_F / ||B_p'||_F × 100  (after projection)."""
        num = float(torch.norm(U_m.float().T @ B_p_proj.float(), p="fro"))
        den = float(torch.norm(B_p_proj.float(), p="fro"))
        return num / (den + 1e-12) * 100.0

    attn_before_all = []
    attn_after_all  = []
    mlp_before_all  = []
    mlp_after_all   = []

    print("\n  ── Layer-wise Leakage Summary ───────────────────────────────")
    print(f"  {'Layer':>5}  {'Attn Before(%)':>14}  {'Attn After(%)':>14}  "
          f"{'MLP Before(%)':>13}  {'MLP After(%)':>12}")
    print(f"  {'-'*5}  {'-'*14}  {'-'*14}  {'-'*13}  {'-'*12}")

    for layer_idx in range(n_layers):
        # ── Attention sub-layers (q, k, v, o) ─────────────────────────────
        attn_before_layer = []
        attn_after_layer  = []
        for sub_idx in range(4):  # q, k, v, o
            seed_base = layer_idx * 100 + sub_idx
            gen = torch.Generator(); gen.manual_seed(seed_base)
            d_out, d_in = attn_shape
            B_m = torch.randn(d_out, r_m, generator=gen)
            A_m = torch.randn(r_m, d_in, generator=gen)
            # Correlated plugin: 50% shared directions
            gen2 = torch.Generator(); gen2.manual_seed(seed_base + 50)
            B_p = torch.randn(d_out, r_p, generator=gen2) + 0.5 * B_m[:, :r_p]
            A_p = torch.randn(r_p, d_in, generator=gen2) + 0.5 * A_m[:r_p, :]

            before = _relative_left_leakage(B_m, B_p)
            attn_before_layer.append(before)

            U_m, _ = compute_subspace_basis(B_m, basis_side="left")
            B_p_proj, A_p_proj = project_two_sided_nsp(B_p, A_p, B_m, A_m)
            after = _relative_left_leakage_after(U_m, B_p_proj)
            attn_after_layer.append(after)

        attn_before_all.append(float(np.mean(attn_before_layer)))
        attn_after_all.append(float(np.mean(attn_after_layer)))

        # ── MLP sub-layers (gate, up, down) ───────────────────────────────
        mlp_before_layer = []
        mlp_after_layer  = []
        for sub_idx, (d_out, d_in) in enumerate(mlp_shapes):
            seed_base = layer_idx * 100 + 10 + sub_idx
            gen = torch.Generator(); gen.manual_seed(seed_base)
            B_m = torch.randn(d_out, r_m, generator=gen)
            A_m = torch.randn(r_m, d_in, generator=gen)
            gen2 = torch.Generator(); gen2.manual_seed(seed_base + 50)
            B_p = torch.randn(d_out, r_p, generator=gen2) + 0.5 * B_m[:, :r_p]
            A_p = torch.randn(r_p, d_in, generator=gen2) + 0.5 * A_m[:r_p, :]

            before = _relative_left_leakage(B_m, B_p)
            mlp_before_layer.append(before)

            U_m, _ = compute_subspace_basis(B_m, basis_side="left")
            B_p_proj, A_p_proj = project_two_sided_nsp(B_p, A_p, B_m, A_m)
            after = _relative_left_leakage_after(U_m, B_p_proj)
            mlp_after_layer.append(after)

        mlp_before_all.append(float(np.mean(mlp_before_layer)))
        mlp_after_all.append(float(np.mean(mlp_after_layer)))

        print(f"  {layer_idx:>5}  {attn_before_all[-1]:>14.4f}  {attn_after_all[-1]:>14.4f}  "
              f"{mlp_before_all[-1]:>13.4f}  {mlp_after_all[-1]:>12.4f}")

    attn_before = np.array(attn_before_all)
    attn_after  = np.array(attn_after_all)
    mlp_before  = np.array(mlp_before_all)
    mlp_after   = np.array(mlp_after_all)

    mean_attn_red = float(np.mean(attn_before) - np.mean(attn_after))
    mean_mlp_red  = float(np.mean(mlp_before)  - np.mean(mlp_after))
    print(f"\n  Avg leakage reduction: Attn = {mean_attn_red:.4f} pp,  MLP = {mean_mlp_red:.4f} pp")
    print(f"  Max residual leakage after NSP: Attn = {attn_after.max():.4f}%,  "
          f"MLP = {mlp_after.max():.4f}%\n")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        "Layer-wise Interference Profile Across 32 Transformer Layers\n"
        f"(LLaMA-3-8B dims: Attn {d_model}×{d_model}, MLP {d_mlp}×{d_model}/{d_model}×{d_mlp}; "
        f"r_m=r_p={r_m})\n"
        "Leakage = ||U_m^T B_p||_F / ||B_p||_F × 100  (before) vs after Asym-NSP",
        fontsize=11, fontweight="bold",
    )

    panel_data = [
        ("Self-Attention Sub-Layers (q/k/v/o avg)", attn_before, attn_after),
        ("SwiGLU MLP Sub-Layers (gate/up/down avg)", mlp_before, mlp_after),
    ]

    x = np.arange(n_layers)
    for ax, (title, before_pct, after_pct) in zip(axes, panel_data):
        ax.fill_between(x, 0, before_pct, alpha=0.15, color=C_NAIVE)
        ax.fill_between(x, 0, after_pct,  alpha=0.25, color=C_NSP)
        ax.plot(x, before_pct, color=C_NAIVE, lw=2.0, marker="o", ms=4,
                label="Before NSP (naive)")
        ax.plot(x, after_pct,  color=C_NSP,   lw=2.0, marker="s", ms=4,
                linestyle="--", label="After Asym-NSP")
        for b in range(0, n_layers, 4):
            ax.axvline(b, color="gray", lw=0.5, alpha=0.4)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_ylabel("Relative Leakage (%)", fontsize=10)
        ax.legend(fontsize=10, loc="upper right")
        ax.set_ylim(-2, max(before_pct.max(), after_pct.max()) * 1.2 + 2)

    axes[-1].set_xlabel("Transformer Layer Index", fontsize=11)
    axes[-1].set_xticks(range(0, n_layers, 2))

    fig.text(
        0.5, -0.01,
        f"Avg leakage reduction: Attn {mean_attn_red:.2f} pp  |  MLP {mean_mlp_red:.2f} pp  "
        f"| Max residual: Attn {attn_after.max():.2f}%  MLP {mlp_after.max():.2f}%",
        ha="center", fontsize=10, style="italic", color="#444",
    )

    plt.tight_layout()
    path = SAVE_DIR / "layerwise_interference_profile.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"     -> Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. MULTI-ADAPTER SCALABILITY  (N = 1 ... 8 plugin adapters)
# ══════════════════════════════════════════════════════════════════════════════

def plot_multi_adapter_capacity():
    print("  [4/4] Generating multi_adapter_capacity.png ...")

    d, r_m, r_p = 4096, 16, 16
    max_adapters = 8

    master_prot_naive = []
    master_prot_nsp   = []
    plugin_energy_nsp = []
    null_space_frac   = []

    # Build master
    torch.manual_seed(0)
    B_m = torch.randn(d, r_m)
    A_m = torch.randn(r_m, d)

    # Cumulative protected bases: start with master only
    protected_B_list = [B_m]
    protected_A_list = [A_m]

    print("\n  ── Multi-Adapter Capacity Summary ───────────────────────────")
    print(f"  {'N':>3}  {'Master(NSP)%':>13}  {'Master(Naive)%':>15}  "
          f"{'Plugin Energy%':>15}  {'Null-Space%':>12}")
    print(f"  {'-'*3}  {'-'*13}  {'-'*15}  {'-'*15}  {'-'*12}")

    for n in range(1, max_adapters + 1):
        torch.manual_seed(n * 7)
        B_p_raw = torch.randn(d, r_p) + 0.5 * B_m[:, :r_p]
        A_p_raw = torch.randn(r_p, d) + 0.5 * A_m[:r_p, :]

        # Build cumulative orthonormal basis from all previously protected B matrices
        U_cum = build_cumulative_subspace_basis(
            protected_B_list, basis_side="left"
        )
        V_cum = build_cumulative_subspace_basis(
            protected_A_list, basis_side="right"
        )

        # Project plugin onto null-space of cumulative protected subspace
        from asym_nsp.core import project_left_null_space, project_right_null_space
        B_p_proj = project_left_null_space(B_p_raw, U_cum)
        A_p_proj = project_right_null_space(A_p_raw, V_cum)

        dW_p_raw  = B_p_raw.float()  @ A_p_raw.float()
        dW_p_proj = B_p_proj.float() @ A_p_proj.float()

        # Errors
        err_naive = _master_output_error(B_m, A_m, B_p_raw, A_p_raw) * 100
        err_nsp   = _master_output_error(B_m, A_m, B_p_proj, A_p_proj) * 100
        eng       = _plugin_energy_sq_pct(dW_p_raw, dW_p_proj)

        # Remaining null-space: (d - r_m - n*r_p) / d  (theoretical lower bound)
        ns_frac = max(0.0, (d - r_m - n * r_p) / d) * 100

        master_prot_naive.append(max(0.0, 100 - err_naive))
        master_prot_nsp.append(100 - err_nsp)
        plugin_energy_nsp.append(eng)
        null_space_frac.append(ns_frac)

        # Add projected matrices to protected lists for next plugin
        protected_B_list.append(B_p_proj)
        protected_A_list.append(A_p_proj)

        print(f"  {n:>3}  {100-err_nsp:>13.4f}  {100-err_naive:>15.4f}  "
              f"{eng:>15.4f}  {ns_frac:>12.4f}")

    print()

    n_vals = np.arange(1, max_adapters + 1)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(
        f"Multi-Adapter Scalability: Cascading N=1…{max_adapters} Plugin Adapters\n"
        f"(d={d}, r_m=r_p={r_m}; U_cum/V_cum via build_cumulative_subspace_basis)",
        fontsize=12, fontweight="bold", y=1.03,
    )

    # Panel A: Master Preservation vs N
    axes[0].plot(n_vals, master_prot_nsp,   color=C_NSP,   lw=2.5, marker="o",
                 label="Asym-NSP (cumulative)")
    axes[0].plot(n_vals, master_prot_naive, color=C_NAIVE, lw=2.0, marker="s",
                 linestyle="--", label="Naive Addition")
    axes[0].fill_between(n_vals, master_prot_nsp, master_prot_naive,
                          alpha=0.12, color=C_NSP)
    axes[0].set_xlabel("Number of Plugin Adapters (N)", fontsize=10)
    axes[0].set_ylabel("Master Preservation (%)", fontsize=10)
    axes[0].set_title("(A) Master Protection vs. N Adapters", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(n_vals)

    # Panel B: Plugin Energy Retained vs N
    bars = axes[1].bar(n_vals, plugin_energy_nsp, color=C_NSP, alpha=0.85, width=0.6,
                       edgecolor="white", linewidth=0.8)
    for n, e in zip(n_vals, plugin_energy_nsp):
        axes[1].text(n, e + 0.3, f"{e:.1f}%", ha="center", fontsize=8, fontweight="bold")
    axes[1].set_xlabel("Number of Plugin Adapters (N)", fontsize=10)
    axes[1].set_ylabel("Plugin Energy Retained (%)\n= ||ΔW_p'||²_F / ||ΔW_p||²_F × 100", fontsize=9)
    axes[1].set_title("(B) Plugin Energy per Adapter vs. N", fontsize=10)
    axes[1].set_ylim(0, max(plugin_energy_nsp) * 1.1 + 5)
    axes[1].set_xticks(n_vals)

    # Panel C: Remaining Null-Space Capacity vs N
    # Formula: (d - r_m - n*r_p) / d × 100  (theoretical, assuming full-rank usage)
    axes[2].fill_between(n_vals, 0, null_space_frac, alpha=0.25, color="#4DAC26")
    axes[2].plot(n_vals, null_space_frac, color="#4DAC26", lw=2.5, marker="D", ms=7)
    for n, ns in zip(n_vals, null_space_frac):
        axes[2].text(n, ns + 0.5, f"{ns:.1f}%", ha="center", fontsize=7)
    axes[2].set_xlabel("Number of Plugin Adapters (N)", fontsize=10)
    axes[2].set_ylabel("Remaining Null-Space\n(% of d dims)", fontsize=10)
    axes[2].set_title(
        f"(C) Null-Space Capacity vs. N\n"
        f"Formula: (d − r_m − N·r_p) / d × 100\n"
        f"d={d}, r_m=r_p={r_m}",
        fontsize=9,
    )
    axes[2].set_ylim(0, 105)
    axes[2].set_xticks(n_vals)

    plt.tight_layout()
    path = SAVE_DIR / "multi_adapter_capacity.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"     -> Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Asymmetric NSP — Full Benchmark Suite Generator")
    print(f"  Output directory: {SAVE_DIR}")
    print("  (All numbers from real measurements — zero hardcoded values)")
    print("=" * 65)
    print()

    t0 = time.perf_counter()
    plot_baseline_comparison()
    plot_rank_ablation_pareto()
    plot_layerwise_interference()
    plot_multi_adapter_capacity()
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 65)
    print(f"  All 4 figures generated in {elapsed:.1f} s")
    print("=" * 65)
    print()
    generated = [
        "baseline_comparison.png",
        "rank_ablation_pareto.png",
        "layerwise_interference_profile.png",
        "multi_adapter_capacity.png",
    ]
    for name in generated:
        fpath = SAVE_DIR / name
        size_kb = fpath.stat().st_size // 1024 if fpath.exists() else 0
        print(f"  * {name:<44}  {size_kb:>5} KB")
    print()


if __name__ == "__main__":
    main()
