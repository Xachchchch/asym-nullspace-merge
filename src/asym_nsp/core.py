"""Mathematical core for Asymmetric Null-Space Projection (NSP).

Provides SVD decomposition, projection operators onto orthogonal complements (null spaces),
subspace basis aggregation for multi-adapter chaining, and subspace preservation metrics
for Low-Rank Adaptation (LoRA) matrices.
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

    basis_side_clean = basis_side.lower()
    if basis_side_clean not in ("left", "right"):
        raise ValueError(f"basis_side must be 'left' or 'right', got {basis_side}")

    dim_out, dim_in = weight.shape
    dim = dim_out if basis_side_clean == "left" else dim_in

    if energy_threshold is not None and not (0.0 <= energy_threshold <= 1.0):
        raise ValueError(
            f"energy_threshold must be between 0.0 and 1.0, got {energy_threshold}"
        )

    orig_dtype = weight.dtype
    # Perform SVD in float32/float64 for numerical precision
    compute_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    w_mat = weight.to(dtype=compute_dtype)

    U, S, Vh = torch.linalg.svd(w_mat, full_matrices=False)
    V = Vh.mH  # [dim_in, num_sv]

    # Handle zero/rank-deficient matrices: filter near-zero singular values
    if S.numel() == 0 or S[0] <= 1e-12:
        return torch.empty((dim, 0), dtype=orig_dtype, device=weight.device), S[:0].to(dtype=orig_dtype)

    eps = 1e-12 * max(weight.shape) * S[0].item()
    active_mask = (S > eps) & (S > 1e-12)
    active_count = int(torch.sum(active_mask).item())
    if active_count == 0:
        return torch.empty((dim, 0), dtype=orig_dtype, device=weight.device), S[:0].to(dtype=orig_dtype)

    U = U[:, :active_count]
    V = V[:, :active_count]
    S = S[:active_count]

    # Determine cutoff rank k
    if energy_threshold is not None:
        if energy_threshold == 0.0:
            k = 0
        elif energy_threshold >= 1.0:
            k = len(S)
            if rank is not None:
                k = min(k, rank)
        else:  # 0.0 < energy_threshold < 1.0
            total_energy = torch.sum(S**2)
            if total_energy > 0:
                cum_energy = torch.cumsum(S**2, dim=0) / total_energy
                matched = (cum_energy >= energy_threshold).nonzero(as_tuple=True)[0]
                k_energy = int(matched[0].item()) + 1 if len(matched) > 0 else len(S)
            else:
                k_energy = 0
            k = min(k_energy, rank) if rank is not None else k_energy
    elif rank is not None:
        k = min(rank, len(S))
    else:
        k = len(S)

    k = max(0, min(k, len(S)))

    if k == 0:
        return torch.empty((dim, 0), dtype=orig_dtype, device=weight.device), S[:0].to(dtype=orig_dtype)

    if basis_side_clean == "left":
        basis = U[:, :k].to(dtype=orig_dtype)
    else:
        basis = V[:, :k].to(dtype=orig_dtype)

    return basis, S[:k].to(dtype=orig_dtype)


def build_cumulative_subspace_basis(
    matrices: list[torch.Tensor],
    rank: Optional[int] = None,
    energy_threshold: Optional[float] = None,
    basis_side: str = "left",
    tol: float = 1e-10,
) -> torch.Tensor:
    """Builds an orthonormal basis spanning the cumulative subspace of multiple matrices.

    Extracts the subspace basis for each matrix in `matrices`, concatenates them along
    the basis column dimension, and runs reduced SVD (keeping singular vectors with
    singular value > tol) in float32/float64 so the returned matrix has strictly
    orthonormal columns spanning the union of all prior subspaces without double-counting
    overlapping directions.

    Args:
        matrices: List of 2D tensors.
        rank: Explicit rank cutoff per matrix.
        energy_threshold: Cumulative energy threshold in [0.0, 1.0] per matrix.
        basis_side: 'left' for output manifold basis, 'right' for input manifold basis.
        tol: Tolerance for discarding near-zero singular values of concatenated basis.

    Returns:
        Orthonormal basis tensor of shape [dim, k_union].
    """
    if not matrices:
        raise ValueError("matrices list must not be empty")

    basis_side_clean = basis_side.lower()
    if basis_side_clean not in ("left", "right"):
        raise ValueError(f"basis_side must be 'left' or 'right', got {basis_side}")

    first_mat = matrices[0]
    if first_mat.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {first_mat.shape}")

    dim = first_mat.shape[0] if basis_side_clean == "left" else first_mat.shape[1]
    orig_dtype = first_mat.dtype
    orig_device = first_mat.device

    bases = []
    for M in matrices:
        if M.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {M.shape}")
        cur_dim = M.shape[0] if basis_side_clean == "left" else M.shape[1]
        if cur_dim != dim:
            raise ValueError(
                f"Dimension mismatch across matrices: expected dimension {dim}, got {cur_dim}"
            )
        basis, _ = compute_subspace_basis(
            M,
            rank=rank,
            energy_threshold=energy_threshold,
            basis_side=basis_side_clean,
        )
        if basis.shape[1] > 0:
            bases.append(basis)

    if not bases:
        return torch.empty((dim, 0), dtype=orig_dtype, device=orig_device)

    combined = torch.cat(bases, dim=1)
    compute_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32

    U_comb, S_comb, _ = torch.linalg.svd(combined.to(dtype=compute_dtype), full_matrices=False)
    keep = S_comb > tol
    num_keep = int(torch.sum(keep).item())

    if num_keep == 0:
        return torch.empty((dim, 0), dtype=orig_dtype, device=orig_device)

    return U_comb[:, :num_keep].to(dtype=orig_dtype, device=orig_device)


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
    if U_master.shape[1] == 0:
        return B_plugin.clone()

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
    if V_master.shape[1] == 0:
        return A_plugin.clone()

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
    """Computes mathematical verification metrics for the projection.

    Uses trace identities to compute Frobenius norms and inner products in O(d * r^2)
    time without constructing dense O(d_out * d_in) weight matrices:
        ||B A||_F^2 = Tr((B^T B)(A A^T))
        <B_m A_m, B_p A_p>_F = Tr((B_m^T B_p)(A_p A_m^T))
    """
    orig_dtype = torch.float64
    b_p = B_plugin_proj.to(dtype=orig_dtype)
    a_p = A_plugin_proj.to(dtype=orig_dtype)
    b_m = B_master.to(dtype=orig_dtype)
    a_m = A_master.to(dtype=orig_dtype)

    U_m, _ = compute_subspace_basis(b_m, energy_threshold=None, basis_side="left")
    V_m, _ = compute_subspace_basis(a_m, energy_threshold=None, basis_side="right")

    b_p_fro = torch.norm(b_p, p="fro")
    a_p_fro = torch.norm(a_p, p="fro")

    if U_m.shape[1] > 0:
        left_leak_mat = torch.matmul(U_m.T, b_p)
        left_leakage_fro = torch.norm(left_leak_mat, p="fro")
        left_subspace_leakage = left_leakage_fro / (b_p_fro + 1e-12)
    else:
        left_leakage_fro = torch.tensor(0.0, dtype=orig_dtype, device=b_p.device)
        left_subspace_leakage = torch.tensor(0.0, dtype=orig_dtype, device=b_p.device)

    if V_m.shape[1] > 0:
        right_leak_mat = torch.matmul(a_p, V_m)
        right_leakage_fro = torch.norm(right_leak_mat, p="fro")
        right_subspace_leakage = right_leakage_fro / (a_p_fro + 1e-12)
    else:
        right_leakage_fro = torch.tensor(0.0, dtype=orig_dtype, device=a_p.device)
        right_subspace_leakage = torch.tensor(0.0, dtype=orig_dtype, device=a_p.device)

    # Compute ||B A||_F^2 = Tr((B^T B)(A A^T)) = sum((B^T B) * (A A^T))
    BtB_m = torch.matmul(b_m.T, b_m)
    AAt_m = torch.matmul(a_m, a_m.T)
    norm_m_sq = torch.sum(BtB_m * AAt_m)

    BtB_p = torch.matmul(b_p.T, b_p)
    AAt_p = torch.matmul(a_p, a_p.T)
    norm_p_sq = torch.sum(BtB_p * AAt_p)

    norm_m = torch.sqrt(torch.clamp(norm_m_sq, min=0.0))
    norm_p = torch.sqrt(torch.clamp(norm_p_sq, min=0.0))

    # <B_m A_m, B_p A_p>_F = Tr((B_m^T B_p)(A_p A_m^T)) = sum((B_m^T B_p) * (A_p A_m^T).T)
    BtB_mp = torch.matmul(b_m.T, b_p)   # [r_m, r_p]
    AAt_pm = torch.matmul(a_p, a_m.T)   # [r_p, r_m]
    inner_prod = torch.sum(BtB_mp * AAt_pm.T)

    interference = inner_prod / (norm_m * norm_p + 1e-12)

    return {
        "left_leakage_fro": float(left_leakage_fro.item()),
        "right_leakage_fro": float(right_leakage_fro.item()),
        "left_subspace_leakage": float(left_subspace_leakage.item()),
        "right_subspace_leakage": float(right_subspace_leakage.item()),
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
