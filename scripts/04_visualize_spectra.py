"""scripts/04_visualize_spectra.py
Generates publication-ready visualizations:
1. Subspace Cosine Interference Heatmap (Before vs. After NSP).
2. Singular Value Energy Spectrum & Principal Angles.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asym_nsp.core import TwoSidedNullSpaceProjector


def compute_principal_angles(A: torch.Tensor, B: torch.Tensor):
    """Computes cosine of principal angles between column spaces of A and B."""
    u_a, _, _ = torch.linalg.svd(A.float(), full_matrices=False)
    u_b, _, _ = torch.linalg.svd(B.float(), full_matrices=False)
    # Cosine of principal angles are singular values of u_a.T @ u_b
    _, cos_angles, _ = torch.linalg.svd(torch.matmul(u_a.T, u_b))
    return cos_angles.numpy()


def generate_visualizations(save_dir: str = "assets"):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Setup Synthetic High-Dimensional LoRAs
    d_out, r_m, r_p = 4096, 16, 16
    torch.manual_seed(42)

    # Create overlapping Master and Plugin
    b_master = torch.randn(d_out, r_m)
    b_plugin_raw = torch.randn(d_out, r_p) + 0.5 * b_master[:, :r_p]  # Correlated/Interfering

    # Apply Asymmetric NSP
    b_plugin_proj, u_master = TwoSidedNullSpaceProjector.project_output_matrix(b_master, b_plugin_raw)

    # -------------------------------------------------------------
    # PLOT 1: Subspace Overlap Heatmap (U_m^T @ B_p)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Before Projection
    overlap_before = (u_master.T @ b_plugin_raw).abs().numpy()
    sns.heatmap(overlap_before, ax=ax1, cmap="YlOrRd", cbar=True)
    ax1.set_title("Before Projection (Raw Overlap)\nHigh Parameter Interference", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Plugin Adapter Rank ($r_p$)")
    ax1.set_ylabel("Master Subspace Directions ($U_m$)")

    # After Projection
    overlap_after = (u_master.T @ b_plugin_proj).abs().numpy()
    sns.heatmap(overlap_after, ax=ax2, cmap="Blues", cbar=True, vmin=0, vmax=1e-6)
    ax2.set_title("After Asymmetric NSP (Ours)\nStrict Zero Interference ($U_m^T B_p' \\approx 0$)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Plugin Adapter Rank ($r_p$)")
    ax2.set_ylabel("Master Subspace Directions ($U_m$)")

    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, "subspace_overlap_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"✓ Saved Heatmap -> {heatmap_path}")

    # -------------------------------------------------------------
    # PLOT 2: Principal Angles & Energy Preservation Spectrum
    # -------------------------------------------------------------
    angles_before = compute_principal_angles(b_master, b_plugin_raw)
    angles_after = compute_principal_angles(b_master, b_plugin_proj)

    plt.figure(figsize=(9, 5))
    x_indices = np.arange(1, len(angles_before) + 1)

    plt.plot(x_indices, angles_before, marker="o", color="crimson", linewidth=2, label="Naive / Unprojected Plugin")
    plt.plot(x_indices, angles_after, marker="s", color="navy", linewidth=2, linestyle="--", label="Asymmetric NSP Plugin (Ours)")

    plt.axhline(0, color="gray", linestyle=":", alpha=0.7)
    plt.title("Cosine of Principal Angles Between Master and Plugin Subspaces", fontsize=13, fontweight="bold")
    plt.xlabel("Principal Angle Index", fontsize=11)
    plt.ylabel("$\\cos(\\theta_i)$  [0 = Orthogonal, 1 = Collinear]", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    spectrum_path = os.path.join(save_dir, "principal_angles_spectrum.png")
    plt.savefig(spectrum_path, dpi=300)
    plt.close()
    print(f"✓ Saved Spectrum Plot -> {spectrum_path}")


if __name__ == "__main__":
    generate_visualizations()
