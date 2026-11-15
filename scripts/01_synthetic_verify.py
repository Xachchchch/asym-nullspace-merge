"""01_synthetic_verify.py
===========================================================
Full reproducible Table 2 verification script for Asymmetric NSP.

Compares:
  1. Naive Addition
  2. Task Arithmetic  (λ = 0.5)
  3. TIES-Merging     (k = 20%, λ = 1.0)
  4. DARE             (p = 0.5, TA λ = 0.5)
  5. Asym-NSP Output-Only   (project_left=True,  project_right=False)
  6. Asym-NSP Two-Sided     (project_left=True,  project_right=True)

Across two distinct regimes:
  • Regime A: Independent Adapters (interference = 0.0)
              (Reproducing Paper Table 2's natural subspace collision)
  • Regime B: High-Conflict Adapters (interference = 0.6)
              (Stress-testing strongly correlated adapters)

Metrics printed per method:
  • Master Subspace Shift (%)    = ||U_m^T (y_merge - y_master)||_F / ||U_m^T y_master||_F * 100
  • Master Full Output Shift (%) = ||y_merge - y_master||_F / ||y_master||_F * 100
  • Left  Leakage ||U_m^T B_p'||_F   (Asym-NSP only; else N/A)
  • Right Leakage ||A_p' V_A||_F     (Asym-NSP only; else N/A)
  • Plugin Energy Retained (%)    = ||ΔW_plugin_out||_F^2 / ||ΔW_p||_F^2 * 100   (Eq. 7)
===========================================================
"""

import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asym_nsp.baselines import (
    merge_dare,
    merge_naive_addition,
    merge_task_arithmetic,
    merge_ties,
)
from asym_nsp.core import compute_subspace_basis, project_two_sided_nsp


D       = 4096
R_M     = 16
R_P     = 16
N_TEST  = 64          # test inputs for shift measurement


def _make_loras(d=D, r_m=R_M, r_p=R_P, interference=0.5, seed=42):
    """Generate synthetic LoRA matrices with controlled interference."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    B_m = torch.randn(d, r_m, generator=gen)
    A_m = torch.randn(r_m, d, generator=gen)
    B_p = torch.randn(d, r_p, generator=gen) + interference * B_m[:, :r_p]
    A_p = torch.randn(r_p, d, generator=gen) + interference * A_m[:r_p, :]
    return B_m, A_m, B_p, A_p


def run_benchmark_regime(
    regime_name: str,
    subtitle: str,
    interference: float,
    seed: int = 42,
) -> Dict[str, dict]:
    """Runs and prints Table 2 comparison for a specific interference regime."""
    print("=" * 110)
    print(f"  {regime_name.upper()}: INTERFERENCE = {interference:.1f}")
    print(f"  ({subtitle})")
    print("=" * 110)

    B_m, A_m, B_p_raw, A_p_raw = _make_loras(interference=interference, seed=seed)

    # ── Master delta and plugin raw delta ─────────────────────────────────────
    dW_m = B_m @ A_m          # [d, d]
    dW_p = B_p_raw @ A_p_raw  # [d, d]

    # ── Master subspace bases for metric computation ───────────────────────────
    U_m, _ = compute_subspace_basis(B_m, basis_side="left")   # [d, r_m]
    V_A, _ = compute_subspace_basis(A_m, basis_side="right")  # [d, r_m]

    # ── Test inputs aligned with Master's input subspace (span of V_A rows) ───
    torch.manual_seed(99)
    coeffs = torch.randn(N_TEST, R_M)
    test_x = coeffs @ V_A.T             # [N_TEST, d]

    # Reference: Master-only output
    y_master = test_x @ dW_m.T          # [N_TEST, d]
    # Master subspace component of reference output
    y_master_sub = y_master @ U_m        # [N_TEST, r_m]

    # ── Frobenius energy of raw plugin delta ──────────────────────────────────
    dW_p_fro_sq = float(torch.sum(dW_p ** 2))

    # ── Helper functions ──────────────────────────────────────────────────────
    def _shift_metrics(dW_plugin_out):
        """Compute subspace and full output shifts relative to Master-only."""
        dW_total = dW_m + dW_plugin_out
        y_merge  = test_x @ dW_total.T          # [N_TEST, d]
        diff     = y_merge - y_master           # [N_TEST, d]
        diff_sub = diff @ U_m                   # [N_TEST, r_m]
        sub_shift  = (torch.norm(diff_sub, p="fro") /
                      (torch.norm(y_master_sub, p="fro") + 1e-12)).item() * 100
        full_shift = (torch.norm(diff, p="fro") /
                      (torch.norm(y_master,     p="fro") + 1e-12)).item() * 100
        return sub_shift, full_shift

    def _plugin_energy_pct(dW_plugin_out):
        """Squared Frobenius energy retained relative to raw plugin delta (Eq. 7)."""
        en = float(torch.sum(dW_plugin_out ** 2))
        return en / (dW_p_fro_sq + 1e-30) * 100.0

    def _left_leakage(B_proj):
        """||U_m^T B_p'||_F  (unnormalized)."""
        return float(torch.norm(U_m.T @ B_proj, p="fro"))

    def _right_leakage(A_proj):
        """||A_p' V_A||_F  (unnormalized)."""
        return float(torch.norm(A_proj @ V_A, p="fro"))

    results = {}

    # 1. Naive Addition
    dW_naive = merge_naive_addition([dW_m, dW_p])
    dW_naive_plugin = dW_naive - dW_m
    sub, full = _shift_metrics(dW_naive_plugin)
    results["Naive Addition"] = dict(
        sub_shift=sub, full_shift=full,
        left_leak="N/A", right_leak="N/A",
        energy_pct=_plugin_energy_pct(dW_naive_plugin),
    )

    # 2. Task Arithmetic (λ=0.5)
    dW_ta = merge_task_arithmetic([dW_m, dW_p], scaling_lam=0.5)
    sub, full = _shift_metrics(dW_ta - dW_m)
    results["Task Arithmetic (λ=0.5)"] = dict(
        sub_shift=sub, full_shift=full,
        left_leak="N/A", right_leak="N/A",
        energy_pct=_plugin_energy_pct(dW_ta - dW_m),
    )

    # 3. TIES-Merging (k=20%, λ=1.0)
    dW_ties = merge_ties([dW_m, dW_p], top_k_pct=20.0, scaling_lam=1.0)
    sub, full = _shift_metrics(dW_ties - dW_m)
    results["TIES-Merging (k=20%, λ=1.0)"] = dict(
        sub_shift=sub, full_shift=full,
        left_leak="N/A", right_leak="N/A",
        energy_pct=_plugin_energy_pct(dW_ties - dW_m),
    )

    # 4. DARE (p=0.5, TA λ=0.5)
    dW_dare = merge_dare([dW_m, dW_p], drop_rate=0.5,
                         base_method="task_arithmetic", scaling_lam=0.5, seed=42)
    sub, full = _shift_metrics(dW_dare - dW_m)
    results["DARE (p=0.5, TA λ=0.5)"] = dict(
        sub_shift=sub, full_shift=full,
        left_leak="N/A", right_leak="N/A",
        energy_pct=_plugin_energy_pct(dW_dare - dW_m),
    )

    # 5. Asym-NSP Output-Only (project_left=True, project_right=False)
    B_nsp_out, A_nsp_out = project_two_sided_nsp(
        B_p_raw, A_p_raw, B_m, A_m,
        project_left=True, project_right=False,
    )
    dW_nsp_out = B_nsp_out @ A_nsp_out
    sub, full = _shift_metrics(dW_nsp_out)
    results["Asym-NSP Output-Only"] = dict(
        sub_shift=sub, full_shift=full,
        left_leak=f"{_left_leakage(B_nsp_out):.4e}",
        right_leak=f"{_right_leakage(A_nsp_out):.4e}",
        energy_pct=_plugin_energy_pct(dW_nsp_out),
    )

    # 6. Asym-NSP Two-Sided (project_left=True, project_right=True)
    B_nsp_2s, A_nsp_2s = project_two_sided_nsp(
        B_p_raw, A_p_raw, B_m, A_m,
        project_left=True, project_right=True,
    )
    dW_nsp_2s = B_nsp_2s @ A_nsp_2s
    sub, full = _shift_metrics(dW_nsp_2s)
    results["Asym-NSP Two-Sided"] = dict(
        sub_shift=sub, full_shift=full,
        left_leak=f"{_left_leakage(B_nsp_2s):.4e}",
        right_leak=f"{_right_leakage(A_nsp_2s):.4e}",
        energy_pct=_plugin_energy_pct(dW_nsp_2s),
    )

    # ── Print results table ───────────────────────────────────────────────────
    col_w = 30
    h_sub   = "Mstr Sub-Shift(%)"
    h_full  = "Mstr Full-Shift(%)"
    h_left  = "Left Leak ||U_m^T B'||"
    h_right = "Right Leak ||A' V_A||"
    h_eng   = "Plugin Energy(%)"

    sep = "-" * 110
    print(sep)
    print(f"{'Method':<{col_w}}  {h_sub:>19}  {h_full:>19}  {h_left:>22}  {h_right:>22}  {h_eng:>16}")
    print(sep)
    for method, m in results.items():
        ll = m["left_leak"] if m["left_leak"] != "N/A" else "        N/A"
        rl = m["right_leak"] if m["right_leak"] != "N/A" else "        N/A"
        print(
            f"{method:<{col_w}}  "
            f"{m['sub_shift']:>19.4f}  "
            f"{m['full_shift']:>19.4f}  "
            f"{ll:>22}  "
            f"{rl:>22}  "
            f"{m['energy_pct']:>16.2f}"
        )
    print(sep)
    print()

    # Invariance assertions
    nsp_2s = results["Asym-NSP Two-Sided"]
    assert nsp_2s["sub_shift"] < 1e-3, \
        f"{regime_name} Two-sided NSP subspace shift too large: {nsp_2s['sub_shift']:.6f}%"
    assert nsp_2s["full_shift"] < 0.2, \
        f"{regime_name} Two-sided NSP full output shift too large: {nsp_2s['full_shift']:.4f}%"

    nsp_out = results["Asym-NSP Output-Only"]
    assert nsp_out["sub_shift"] < 1e-3, \
        f"{regime_name} Output-only NSP subspace shift too large: {nsp_out['sub_shift']:.6f}%"

    return results


def run_verification():
    print("=" * 110)
    print("  ASYMMETRIC NULL-SPACE PROJECTION (NSP) — SYNTHETIC BENCHMARK SUITE")
    print(f"  Configuration: d = {D},  r_m = {R_M},  r_p = {R_P},  dtype = float32")
    print("=" * 110)
    print()

    # Regime A: Independent Adapters (interference = 0.0)
    run_benchmark_regime(
        regime_name="Regime A: Independent Adapters",
        subtitle="Reproducing Paper Table 2's Natural Subspace Collision",
        interference=0.0,
        seed=42,
    )

    # Regime B: High-Conflict Adapters (interference = 0.6)
    run_benchmark_regime(
        regime_name="Regime B: High-Conflict Adapters",
        subtitle="Stress-Testing Strongly Correlated Adapters",
        interference=0.6,
        seed=42,
    )

    print("Legend:")
    print("  Mstr Sub-Shift(%)   = ||U_m^T(y_merge - y_master)||_F / ||U_m^T y_master||_F * 100")
    print("  Mstr Full-Shift(%)  = ||y_merge - y_master||_F / ||y_master||_F * 100")
    print("  Left Leak           = ||U_m^T B_p'||_F  (Asym-NSP only)")
    print("  Right Leak          = ||A_p' V_A||_F    (Asym-NSP only)")
    print("  Plugin Energy(%)    = ||ΔW_plugin_out||_F^2 / ||ΔW_p||_F^2 * 100  (Eq. 7)")
    print("  For symmetric baselines: plugin_out = ΔW_merged − ΔW_m\n")

    print("✓ All mathematical invariance checks passed for both regimes.\n")
    print("=" * 110)


if __name__ == "__main__":
    run_verification()
