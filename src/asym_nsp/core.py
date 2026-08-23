"""Mathematical core for Asymmetric Null-Space Projection (NSP).

Provides SVD decomposition, projection operators onto orthogonal complements (null spaces),
and subspace preservation metrics for Low-Rank Adaptation (LoRA) matrices.
"""

from typing import Optional, Tuple, Union
import torch


def compute_subspace_basis(
    weight: torch.Tensor,
    rank: Optional[int] = None,
    energy_threshold: Optional[float] = None,
    basis_side: str = "left",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes orthonormal basis vectors spanning the active subspace of a weight matrix.

    Args:
        weight: Input 2D tensor [dim_out, dim_in].
        rank: Explicit rank cutoff k. If None, determined by energy_threshold or full rank.
        energy_threshold: Fraction of variance/energy to retain in [0.0, 1.0] (e.g. 0.99).
        basis_side: 'left' to return U basis [dim_out, k], 'right' to return V basis [dim_in, k].

    Returns:
        Tuple of (basis_matrix, singular_values).
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {weight.shape}")

    orig_dtype = weight.dtype
    # Perform SVD in float32/float64 for numerical precision
    compute_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    w_mat = weight.to(dtype=compute_dtype)

    U, S, Vh = torch.linalg.svd(w_mat, full_matrices=False)
    V = Vh.mH  # [dim_in, num_sv]

    # Determine cutoff rank k
    max_k = S.shape[0]
    if energy_threshold is not None and 0.0 < energy_threshold < 1.0:
        total_energy = torch.sum(S**2)
        if total_energy > 0:
            cum_energy = torch.cumsum(S**2, dim=0) / total_energy
            k_energy = int((cum_energy >= energy_threshold).nonzero(as_tuple=True)[0][0].item()) + 1
        else:
            k_energy = 1
        k = min(k_energy, rank) if rank is not None else k_energy
    elif rank is not None:
        k = min(rank, max_k)
    else:
        k = max_k

    k = max(1, min(k, max_k))

    if basis_side.lower() == "left":
        basis = U[:, :k].to(dtype=orig_dtype)
    elif basis_side.lower() == "right":
        basis = V[:, :k].to(dtype=orig_dtype)
    else:
        raise ValueError(f"basis_side must be 'left' or 'right', got {basis_side}")

    return basis, S[:k].to(dtype=orig_dtype)


def project_left_null_space(
    B_plugin: torch.Tensor,
    U_master: torch.Tensor,
) -> torch.Tensor:
    """Projects B_plugin onto the left null-space of Master adapter.

    Computes: B_plugin' = (I - U_master @ U_master.T) @ B_plugin
                        = B_plugin - U_master @ (U_master.T @ B_plugin)
    Time Complexity: O(d_out * k * r_plugin), avoiding explicit (d_out x d_out) matrix.

    Args:
        B_plugin: Plugin B matrix of shape [d_out, r_p]
        U_master: Orthonormal basis of Master B of shape [d_out, k]

    Returns:
        Projected B_plugin' of shape [d_out, r_p]
    """
    if B_plugin.ndim != 2 or U_master.ndim != 2:
        raise ValueError("Inputs must be 2D tensors")
    if B_plugin.shape[0] != U_master.shape[0]:
        raise ValueError(
            f"Dimension mismatch: B_plugin {B_plugin.shape} vs U_master {U_master.shape}"
        )

    orig_dtype = B_plugin.dtype
    compute_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32

    B_comp = B_plugin.to(dtype=compute_dtype)
    U_comp = U_master.to(dtype=compute_dtype)

    # Orthogonal projection: B' = B - U (U^T B)
    overlap = torch.matmul(U_comp.T, B_comp)  # [k, r_p]
    projected = B_comp - torch.matmul(U_comp, overlap)  # [d_out, r_p]

    return projected.to(dtype=orig_dtype)


def project_right_null_space(
    A_plugin: torch.Tensor,
    V_master: torch.Tensor,
) -> torch.Tensor:
    """Projects A_plugin onto the right null-space of Master adapter.

    Computes: A_plugin' = A_plugin @ (I - V_master @ V_master.T)
                        = A_plugin - (A_plugin @ V_master) @ V_master.T
    Time Complexity: O(d_in * k * r_plugin), avoiding explicit (d_in x d_in) matrix.

    Args:
        A_plugin: Plugin A matrix of shape [r_p, d_in]
        V_master: Orthonormal basis of Master A of shape [d_in, k]

    Returns:
        Projected A_plugin' of shape [r_p, d_in]
    """
    if A_plugin.ndim != 2 or V_master.ndim != 2:
        raise ValueError("Inputs must be 2D tensors")
    if A_plugin.shape[1] != V_master.shape[0]:
        raise ValueError(
            f"Dimension mismatch: A_plugin {A_plugin.shape} vs V_master {V_master.shape}"
        )

    orig_dtype = A_plugin.dtype
    compute_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32

    A_comp = A_plugin.to(dtype=compute_dtype)
    V_comp = V_master.to(dtype=compute_dtype)

    # Orthogonal projection: A' = A - (A V) V^T
    overlap = torch.matmul(A_comp, V_comp)  # [r_p, k]
    projected = A_comp - torch.matmul(overlap, V_comp.T)  # [r_p, d_in]

    return projected.to(dtype=orig_dtype)


def project_two_sided_nsp(
    B_plugin: torch.Tensor,
    A_plugin: torch.Tensor,
    B_master: torch.Tensor,
    A_master: torch.Tensor,
    rank_k: Optional[int] = None,
    energy_threshold: Optional[float] = None,
    project_left: bool = True,
    project_right: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Two-Sided Asymmetric Null-Space Projection to plugin LoRA pair (B_p, A_p).

    Args:
        B_plugin: Plugin B matrix [d_out, r_p]
        A_plugin: Plugin A matrix [r_p, d_in]
        B_master: Master B matrix [d_out, r_m]
        A_master: Master A matrix [r_m, d_in]
        rank_k: Optional maximum rank cutoff for projector basis
        energy_threshold: Optional cumulative energy threshold in (0, 1]
        project_left: Whether to project output manifold (B_plugin onto null(B_master))
        project_right: Whether to project input manifold (A_plugin onto null(A_master))

    Returns:
        Tuple (B_plugin_projected, A_plugin_projected)
    """
    B_proj = B_plugin
    A_proj = A_plugin

    if project_left:
        U_master, _ = compute_subspace_basis(
            B_master,
            rank=rank_k,
            energy_threshold=energy_threshold,
            basis_side="left",
        )
        B_proj = project_left_null_space(B_plugin, U_master)

    if project_right:
        V_master, _ = compute_subspace_basis(
            A_master,
            rank=rank_k,
            energy_threshold=energy_threshold,
            basis_side="right",
        )
        A_proj = project_right_null_space(A_plugin, V_master)

    return B_proj, A_proj


def compute_orthogonality_metrics(
    B_plugin_proj: torch.Tensor,
    A_plugin_proj: torch.Tensor,
    B_master: torch.Tensor,
    A_master: torch.Tensor,
) -> dict:
    """Computes mathematical verification metrics for the projection."""
    orig_dtype = torch.float64
    b_p = B_plugin_proj.to(dtype=orig_dtype)
    a_p = A_plugin_proj.to(dtype=orig_dtype)
    b_m = B_master.to(dtype=orig_dtype)
    a_m = A_master.to(dtype=orig_dtype)

    U_m, _ = compute_subspace_basis(b_m, basis_side="left")
    V_m, _ = compute_subspace_basis(a_m, basis_side="right")

    left_leakage = torch.norm(torch.matmul(U_m.T, b_p), p="fro") / (torch.norm(b_p, p="fro") + 1e-12)
    right_leakage = torch.norm(torch.matmul(a_p, V_m), p="fro") / (torch.norm(a_p, p="fro") + 1e-12)

    delta_w_master = torch.matmul(b_m, a_m)
    delta_w_plugin = torch.matmul(b_p, a_p)

    # Interference: Frobenius inner product / overlap
    interference = torch.sum(delta_w_master * delta_w_plugin) / (
        torch.norm(delta_w_master, p="fro") * torch.norm(delta_w_plugin, p="fro") + 1e-12
    )

    return {
        "left_subspace_leakage": float(left_leakage.item()),
        "right_subspace_leakage": float(right_leakage.item()),
        "subspace_interference_cosine": float(interference.item()),
    }


class TwoSidedNullSpaceProjector:
    """Convenience class providing static projection methods for input/output manifolds."""

    @staticmethod
    def project_output_matrix(
        B_master: torch.Tensor,
        B_plugin: torch.Tensor,
        rank_k: Optional[int] = None,
        energy_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projects B_plugin onto the left null-space of B_master.

        Returns:
            Tuple of (B_plugin_projected, U_master_basis).
        """
        U_master, _ = compute_subspace_basis(
            B_master,
            rank=rank_k,
            energy_threshold=energy_threshold,
            basis_side="left",
        )
        B_proj = project_left_null_space(B_plugin, U_master)
        return B_proj, U_master

    @staticmethod
    def project_input_matrix(
        A_master: torch.Tensor,
        A_plugin: torch.Tensor,
        rank_k: Optional[int] = None,
        energy_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projects A_plugin onto the right null-space of A_master.

        Returns:
            Tuple of (A_plugin_projected, V_master_basis).
        """
        V_master, _ = compute_subspace_basis(
            A_master,
            rank=rank_k,
            energy_threshold=energy_threshold,
            basis_side="right",
        )
        A_proj = project_right_null_space(A_plugin, V_master)
        return A_proj, V_master

    @staticmethod
    def project(
        B_master: torch.Tensor,
        A_master: torch.Tensor,
        B_plugin: torch.Tensor,
        A_plugin: torch.Tensor,
        rank_k: Optional[int] = None,
        energy_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies two-sided null space projection."""
        return project_two_sided_nsp(
            B_plugin=B_plugin,
            A_plugin=A_plugin,
            B_master=B_master,
            A_master=A_master,
            rank_k=rank_k,
            energy_threshold=energy_threshold,
        )
