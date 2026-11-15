"""Unit tests for mathematical orthogonality guarantees in Asymmetric NSP."""

import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asym_nsp.core import (
    build_cumulative_subspace_basis,
    compute_orthogonality_metrics,
    compute_subspace_basis,
    project_left_null_space,
    project_right_null_space,
    project_two_sided_nsp,
)


class TestOrthogonality(unittest.TestCase):

    def test_left_null_space_projection(self):
        """Verifies that U_m^T @ B_p' == 0."""
        torch.manual_seed(42)
        d_out, r_m, r_p = 256, 16, 8

        B_m = torch.randn(d_out, r_m, dtype=torch.float64)
        B_p = torch.randn(d_out, r_p, dtype=torch.float64)

        U_m, _ = compute_subspace_basis(B_m, basis_side="left")
        B_p_proj = project_left_null_space(B_p, U_m)

        overlap = torch.matmul(U_m.T, B_p_proj)
        residual_norm = torch.norm(overlap, p="fro").item()

        self.assertLess(residual_norm, 1e-10, f"Left null-space projection failed, residual: {residual_norm}")

    def test_right_null_space_projection(self):
        """Verifies that A_p' @ V_m == 0."""
        torch.manual_seed(42)
        d_in, r_m, r_p = 512, 16, 8

        A_m = torch.randn(r_m, d_in, dtype=torch.float64)
        A_p = torch.randn(r_p, d_in, dtype=torch.float64)

        V_m, _ = compute_subspace_basis(A_m, basis_side="right")
        A_p_proj = project_right_null_space(A_p, V_m)

        overlap = torch.matmul(A_p_proj, V_m)
        residual_norm = torch.norm(overlap, p="fro").item()

        self.assertLess(residual_norm, 1e-10, f"Right null-space projection failed, residual: {residual_norm}")

    def test_two_sided_nsp_metrics(self):
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

        self.assertLess(metrics["left_subspace_leakage"], 1e-5)
        self.assertLess(metrics["right_subspace_leakage"], 1e-5)
        self.assertIn("left_leakage_fro", metrics)
        self.assertIn("right_leakage_fro", metrics)
        self.assertIn("subspace_interference_cosine", metrics)

    def test_idempotence(self):
        """Verifies that projecting twice produces the exact same projected matrix."""
        torch.manual_seed(42)
        d_out, r_m, r_p = 64, 8, 8

        B_m = torch.randn(d_out, r_m, dtype=torch.float64)
        B_p = torch.randn(d_out, r_p, dtype=torch.float64)

        U_m, _ = compute_subspace_basis(B_m, basis_side="left")
        B_proj_1 = project_left_null_space(B_p, U_m)
        B_proj_2 = project_left_null_space(B_proj_1, U_m)

        diff = torch.norm(B_proj_1 - B_proj_2, p="fro").item()
        self.assertLess(diff, 1e-12)

    def test_zero_matrix_master(self):
        """Verifies that zero B_master / A_master leaves B_plugin / A_plugin unchanged."""
        torch.manual_seed(42)
        d_out, d_in, r_m, r_p = 128, 64, 16, 8

        B_m = torch.zeros(d_out, r_m, dtype=torch.float64)
        A_m = torch.zeros(r_m, d_in, dtype=torch.float64)
        B_p = torch.randn(d_out, r_p, dtype=torch.float64)
        A_p = torch.randn(r_p, d_in, dtype=torch.float64)

        # Basis of zero matrix must have 0 columns
        U_m, S_m = compute_subspace_basis(B_m, basis_side="left")
        self.assertEqual(U_m.shape, (d_out, 0))
        self.assertEqual(S_m.shape, (0,))

        V_m, S_vm = compute_subspace_basis(A_m, basis_side="right")
        self.assertEqual(V_m.shape, (d_in, 0))
        self.assertEqual(S_vm.shape, (0,))

        # Directly testing projection functions
        B_p_proj_direct = project_left_null_space(B_p, U_m)
        self.assertTrue(torch.equal(B_p_proj_direct, B_p))

        A_p_proj_direct = project_right_null_space(A_p, V_m)
        self.assertTrue(torch.equal(A_p_proj_direct, A_p))

        # Testing full two-sided nsp
        B_p_proj, A_p_proj = project_two_sided_nsp(
            B_plugin=B_p,
            A_plugin=A_p,
            B_master=B_m,
            A_master=A_m,
        )
        self.assertTrue(torch.equal(B_p_proj, B_p))
        self.assertTrue(torch.equal(A_p_proj, A_p))

    def test_energy_threshold_edge_cases(self):
        """Verifies energy_threshold validation and edge cases 0.0 and 1.0."""
        torch.manual_seed(42)
        d_out, d_in = 64, 32
        W = torch.randn(d_out, d_in, dtype=torch.float64)

        # Validation of range [0.0, 1.0]
        with self.assertRaises(ValueError):
            compute_subspace_basis(W, energy_threshold=-0.01)
        with self.assertRaises(ValueError):
            compute_subspace_basis(W, energy_threshold=1.01)

        # energy_threshold == 0.0: must return empty basis [dim, 0]
        U_0, S_0 = compute_subspace_basis(W, energy_threshold=0.0, basis_side="left")
        self.assertEqual(U_0.shape, (d_out, 0))
        self.assertEqual(S_0.shape, (0,))

        V_0, S_v0 = compute_subspace_basis(W, energy_threshold=0.0, basis_side="right")
        self.assertEqual(V_0.shape, (d_in, 0))
        self.assertEqual(S_v0.shape, (0,))

        # energy_threshold == 1.0: must keep all active singular vectors
        U_1, S_1 = compute_subspace_basis(W, energy_threshold=1.0, basis_side="left")
        self.assertEqual(U_1.shape, (d_out, min(d_out, d_in)))
        self.assertEqual(S_1.shape, (min(d_out, d_in),))

        # energy_threshold == 1.0 capped by rank
        rank_cap = 5
        U_1_cap, S_1_cap = compute_subspace_basis(
            W, rank=rank_cap, energy_threshold=1.0, basis_side="left"
        )
        self.assertEqual(U_1_cap.shape, (d_out, rank_cap))
        self.assertEqual(S_1_cap.shape, (rank_cap,))

        # 0.0 < energy_threshold < 1.0
        U_90, _ = compute_subspace_basis(W, energy_threshold=0.90, basis_side="left")
        self.assertGreater(U_90.shape[1], 0)
        self.assertLessEqual(U_90.shape[1], min(d_out, d_in))

    def test_build_cumulative_subspace_basis(self):
        """Verifies cumulative basis over 3 correlated matrices has Q^T Q ≈ I and zero leakage."""
        torch.manual_seed(42)
        d_out, d_in = 64, 16
        r_shared = 4

        # Left basis test: 3 correlated matrices sharing a 4-dimensional left subspace
        base_U, _ = torch.linalg.qr(torch.randn(d_out, r_shared, dtype=torch.float64))
        m1 = base_U @ torch.randn(r_shared, d_in, dtype=torch.float64)
        m2 = base_U @ torch.randn(r_shared, d_in, dtype=torch.float64)
        m3 = base_U @ torch.randn(r_shared, d_in, dtype=torch.float64)

        Q_left = build_cumulative_subspace_basis([m1, m2, m3], basis_side="left", tol=1e-10)

        # 1. Orthonormality check: Q^T Q ≈ I
        self.assertEqual(Q_left.shape, (d_out, r_shared))
        I_left = torch.eye(Q_left.shape[1], dtype=torch.float64)
        ortho_err_left = torch.norm(torch.matmul(Q_left.T, Q_left) - I_left, p="fro").item()
        self.assertLess(ortho_err_left, 1e-12, f"Left cumulative basis not orthonormal: {ortho_err_left}")

        # 2. Zero leakage against all 3 input matrices: (I - Q Q^T) M == 0
        for idx, m in enumerate([m1, m2, m3]):
            proj_left = project_left_null_space(m, Q_left)
            leakage = torch.norm(proj_left, p="fro").item()
            self.assertLess(
                leakage, 1e-10, f"Left leakage on matrix {idx+1} is non-zero: {leakage}"
            )

        # Right basis test: 3 correlated matrices sharing a 4-dimensional right subspace
        base_V, _ = torch.linalg.qr(torch.randn(d_in, r_shared, dtype=torch.float64))
        m1_r = torch.randn(d_out, r_shared, dtype=torch.float64) @ base_V.T
        m2_r = torch.randn(d_out, r_shared, dtype=torch.float64) @ base_V.T
        m3_r = torch.randn(d_out, r_shared, dtype=torch.float64) @ base_V.T

        Q_right = build_cumulative_subspace_basis([m1_r, m2_r, m3_r], basis_side="right", tol=1e-10)

        # 1. Orthonormality check: Q^T Q ≈ I
        self.assertEqual(Q_right.shape, (d_in, r_shared))
        I_right = torch.eye(Q_right.shape[1], dtype=torch.float64)
        ortho_err_right = torch.norm(torch.matmul(Q_right.T, Q_right) - I_right, p="fro").item()
        self.assertLess(ortho_err_right, 1e-12, f"Right cumulative basis not orthonormal: {ortho_err_right}")

        # 2. Zero leakage against all 3 input matrices: M (I - Q Q^T) == 0
        for idx, m in enumerate([m1_r, m2_r, m3_r]):
            proj_right = project_right_null_space(m, Q_right)
            leakage_r = torch.norm(proj_right, p="fro").item()
            self.assertLess(
                leakage_r, 1e-10, f"Right leakage on matrix {idx+1} is non-zero: {leakage_r}"
            )

    def test_subspace_interference_cosine_trace_identity(self):
        """Verifies O(d r^2) subspace_interference_cosine matches naive torch.matmul to < 1e-10."""
        torch.manual_seed(999)
        d_out, d_in = 256, 128
        r_m, r_p = 16, 8

        B_m = torch.randn(d_out, r_m, dtype=torch.float64)
        A_m = torch.randn(r_m, d_in, dtype=torch.float64)
        B_p = torch.randn(d_out, r_p, dtype=torch.float64)
        A_p = torch.randn(r_p, d_in, dtype=torch.float64)

        metrics = compute_orthogonality_metrics(
            B_plugin_proj=B_p,
            A_plugin_proj=A_p,
            B_master=B_m,
            A_master=A_m,
        )

        cosine_fast = metrics["subspace_interference_cosine"]

        # Naive dense O(d_out * d_in) inner product and norms
        delta_w_m = torch.matmul(B_m, A_m)
        delta_w_p = torch.matmul(B_p, A_p)

        inner_prod_naive = torch.sum(delta_w_m * delta_w_p)
        norm_m_naive = torch.norm(delta_w_m, p="fro")
        norm_p_naive = torch.norm(delta_w_p, p="fro")
        cosine_naive = (inner_prod_naive / (norm_m_naive * norm_p_naive + 1e-12)).item()

        diff = abs(cosine_fast - cosine_naive)
        self.assertLess(
            diff,
            1e-10,
            f"Trace identity cosine {cosine_fast} differs from naive {cosine_naive} by {diff}",
        )

        # Check that Table 2 metrics are also present and non-negative
        self.assertIn("left_leakage_fro", metrics)
        self.assertIn("right_leakage_fro", metrics)
        self.assertGreaterEqual(metrics["left_leakage_fro"], 0.0)
        self.assertGreaterEqual(metrics["right_leakage_fro"], 0.0)


if __name__ == "__main__":
    unittest.main()
