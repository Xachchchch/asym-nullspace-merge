"""Unit tests for mathematical orthogonality guarantees in Asymmetric NSP."""

import pytest
import torch

from asym_nsp.core import (
    compute_orthogonality_metrics,
    compute_subspace_basis,
    project_left_null_space,
    project_right_null_space,
    project_two_sided_nsp,
)


def test_left_null_space_projection():
    """Verifies that U_m^T @ B_p' == 0."""
    torch.manual_seed(42)
    d_out, r_m, r_p = 256, 16, 8

    # Generate synthetic random matrices
    B_m = torch.randn(d_out, r_m, dtype=torch.float64)
    B_p = torch.randn(d_out, r_p, dtype=torch.float64)

    U_m, _ = compute_subspace_basis(B_m, basis_side="left")
    B_p_proj = project_left_null_space(B_p, U_m)

    # Compute overlap U_m^T @ B_p'
    overlap = torch.matmul(U_m.T, B_p_proj)
    residual_norm = torch.norm(overlap, p="fro").item()

    assert residual_norm < 1e-10, f"Left null-space projection failed, residual: {residual_norm}"


def test_right_null_space_projection():
    """Verifies that A_p' @ V_m == 0."""
    torch.manual_seed(42)
    d_in, r_m, r_p = 512, 16, 8

    A_m = torch.randn(r_m, d_in, dtype=torch.float64)
    A_p = torch.randn(r_p, d_in, dtype=torch.float64)

    V_m, _ = compute_subspace_basis(A_m, basis_side="right")
    A_p_proj = project_right_null_space(A_p, V_m)

    # Compute overlap A_p' @ V_m
    overlap = torch.matmul(A_p_proj, V_m)
    residual_norm = torch.norm(overlap, p="fro").item()

    assert residual_norm < 1e-10, f"Right null-space projection failed, residual: {residual_norm}"


def test_two_sided_nsp_metrics():
    """Verifies two-sided projection metrics and subspace interference reduction."""
    torch.manual_seed(123)
    d_out, d_in = 128, 128
    r_m, r_p = 16, 16

    B_m = torch.randn(d_out, r_m, dtype=torch.float32)
    A_m = torch.randn(r_m, d_in, dtype=torch.float32)
    B_p = torch.randn(d_out, r_p, dtype=torch.float32)
    A_p = torch.randn(r_p, d_in, dtype=torch.float32)

    B_p_proj, A_p_proj = project_two_sided_nsp(
        B_plugin=B_p,
        A_plugin=A_p,
        B_master=B_m,
        A_master=A_m,
    )

    metrics = compute_orthogonality_metrics(
        B_plugin_proj=B_p_proj,
        A_plugin_proj=A_p_proj,
        B_master=B_m,
        A_master=A_m,
    )

    assert metrics["left_subspace_leakage"] < 1e-5
    assert metrics["right_subspace_leakage"] < 1e-5


def test_idempotence():
    """Verifies that projecting twice produces the exact same projected matrix."""
    torch.manual_seed(42)
    d_out, r_m, r_p = 64, 8, 8

    B_m = torch.randn(d_out, r_m, dtype=torch.float64)
    B_p = torch.randn(d_out, r_p, dtype=torch.float64)

    U_m, _ = compute_subspace_basis(B_m, basis_side="left")
    B_proj_1 = project_left_null_space(B_p, U_m)
    B_proj_2 = project_left_null_space(B_proj_1, U_m)

    diff = torch.norm(B_proj_1 - B_proj_2, p="fro").item()
    assert diff < 1e-12
