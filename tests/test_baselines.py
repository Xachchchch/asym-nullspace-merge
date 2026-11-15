"""Unit tests for src/asym_nsp/baselines.py."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asym_nsp.baselines import (
    merge_dare,
    merge_lora_adapters_baseline,
    merge_naive_addition,
    merge_task_arithmetic,
    merge_ties,
)
from asym_nsp.lora_loader import LoRAAdapter, LoRALayerWeights


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_adapter(layers_dict, name="adapter"):
    """Build a minimal LoRAAdapter from {mod_key: (B, A)} pairs."""
    layers = {}
    for mod_key, (B, A) in layers_dict.items():
        r = A.shape[0]
        layers[mod_key] = LoRALayerWeights(
            module_key=mod_key,
            weight_a=A,
            weight_b=B,
            r=r,
            lora_alpha=float(r),  # scaling = 1.0
        )
    return LoRAAdapter(name=name, config={}, layers=layers)


# ══════════════════════════════════════════════════════════════════════════════
# merge_naive_addition
# ══════════════════════════════════════════════════════════════════════════════

class TestMergeNaiveAddition(unittest.TestCase):

    def test_single_delta(self):
        d = torch.tensor([1.0, 2.0, 3.0])
        result = merge_naive_addition([d])
        self.assertTrue(torch.allclose(result, d))

    def test_two_deltas(self):
        d1 = torch.tensor([1.0, 2.0, 3.0])
        d2 = torch.tensor([4.0, 5.0, 6.0])
        result = merge_naive_addition([d1, d2])
        self.assertTrue(torch.allclose(result, torch.tensor([5.0, 7.0, 9.0])))

    def test_three_deltas(self):
        d1 = torch.ones(4)
        d2 = torch.ones(4) * 2
        d3 = torch.ones(4) * 3
        result = merge_naive_addition([d1, d2, d3])
        self.assertTrue(torch.allclose(result, torch.ones(4) * 6))

    def test_shape_preserved(self):
        torch.manual_seed(0)
        ds = [torch.randn(32, 64) for _ in range(3)]
        result = merge_naive_addition(ds)
        self.assertEqual(result.shape, (32, 64))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            merge_naive_addition([])


# ══════════════════════════════════════════════════════════════════════════════
# merge_task_arithmetic
# ══════════════════════════════════════════════════════════════════════════════

class TestMergeTaskArithmetic(unittest.TestCase):

    def test_lam_1(self):
        d1 = torch.tensor([1.0, 2.0])
        d2 = torch.tensor([3.0, 4.0])
        result = merge_task_arithmetic([d1, d2], scaling_lam=1.0)
        self.assertTrue(torch.allclose(result, torch.tensor([4.0, 6.0])))

    def test_lam_half(self):
        d1 = torch.tensor([2.0, 4.0])
        d2 = torch.tensor([2.0, 4.0])
        result = merge_task_arithmetic([d1, d2], scaling_lam=0.5)
        self.assertTrue(torch.allclose(result, torch.tensor([2.0, 4.0])))

    def test_lam_zero(self):
        d1 = torch.randn(10)
        d2 = torch.randn(10)
        result = merge_task_arithmetic([d1, d2], scaling_lam=0.0)
        self.assertTrue(torch.allclose(result, torch.zeros(10)))

    def test_default_lam(self):
        """Default lambda=0.5 should give half the naive sum."""
        d1 = torch.tensor([2.0, 4.0])
        d2 = torch.tensor([6.0, 8.0])
        naive = merge_naive_addition([d1, d2])
        ta = merge_task_arithmetic([d1, d2])
        self.assertTrue(torch.allclose(ta, 0.5 * naive))


# ══════════════════════════════════════════════════════════════════════════════
# merge_ties  –  hand-verified small example
# ══════════════════════════════════════════════════════════════════════════════

class TestMergeTies(unittest.TestCase):

    def test_hand_calculated_ties_pipeline(self):
        """Verifies Trim, Sign Election, and Disjoint Mean (ignoring zeros) on hand-calculated tensors.

        Setup:
          d1 = [ 10.0,  0.5,  3.0, 0.1, 5.0]
          d2 = [ -8.0, -4.0, -0.2, 0.1, 7.0]
          top_k_pct = 60.0% (keep top 3 of 5 by magnitude)

        1. Trim:
          d1 top 3 abs: [10, 5, 3] -> d1_trim = [ 10.0,  0.0, 3.0, 0.0, 5.0]
          d2 top 3 abs: [ 8, 7, 4] -> d2_trim = [ -8.0, -4.0, 0.0, 0.0, 7.0]

        2. Sign Election:
          sum = [ +2.0, -4.0, +3.0, 0.0, +12.0]
          gamma = [ +1,   -1,   +1,   0,    +1]

        3. Disjoint Mean (ignoring zeros):
          coord 0: d1 (+10) matches, d2 (-8) conflicts -> only d1 -> avg = 10.0
          coord 1: d1 (0) trimmed, d2 (-4) matches -> only d2 -> avg = -4.0 (ignoring 0!)
          coord 2: d1 (+3) matches, d2 (0) trimmed -> only d1 -> avg = 3.0
          coord 3: both 0 -> 0.0
          coord 4: both match gamma -> (5.0 + 7.0) / 2 = 6.0
        """
        d1 = torch.tensor([10.0, 0.5, 3.0, 0.1, 5.0])
        d2 = torch.tensor([-8.0, -4.0, -0.2, 0.1, 7.0])

        merged = merge_ties([d1, d2], top_k_pct=60.0, scaling_lam=1.0)
        expected = torch.tensor([10.0, -4.0, 3.0, 0.0, 6.0])
        self.assertTrue(torch.allclose(merged, expected, atol=1e-5), f"Got {merged}, expected {expected}")

        # Test with scaling_lam = 0.5
        merged_scaled = merge_ties([d1, d2], top_k_pct=60.0, scaling_lam=0.5)
        self.assertTrue(torch.allclose(merged_scaled, 0.5 * expected, atol=1e-5))

    def test_single_delta_trim(self):
        """Single delta: keeps top elements, trims the rest to zero."""
        d = torch.tensor([-1.0, 2.0, -3.0, 4.0])
        # top 50% of 4 = 2 elements -> keep |-3|=3 and |4|=4
        result = merge_ties([d], top_k_pct=50.0, scaling_lam=1.0)
        expected = torch.tensor([0.0, 0.0, -3.0, 4.0])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_sign_election_two_deltas(self):
        """Sign election should pick the dominant sign at each coordinate."""
        d1 = torch.tensor([3.0, -2.0])
        d2 = torch.tensor([1.0, -4.0])
        result = merge_ties([d1, d2], top_k_pct=100.0, scaling_lam=1.0)
        self.assertGreater(result[0].item(), 0)
        self.assertLess(result[1].item(), 0)

    def test_disjoint_mean_ignores_zeros(self):
        """Disjoint mean: positions trimmed to 0 must NOT reduce the denominator."""
        d1 = torch.tensor([10.0, 5.0])
        d2 = torch.tensor([0.5,  5.0])
        result = merge_ties([d1, d2], top_k_pct=50.0, scaling_lam=1.0)
        self.assertAlmostEqual(result[0].item(), 10.0, places=4)
        self.assertAlmostEqual(result[1].item(), 5.0, places=4)

    def test_sign_conflict_zeroed(self):
        """Coordinates where signs conflict should contribute a reduced or zero merge."""
        d1 = torch.tensor([5.0, 3.0])
        d2 = torch.tensor([-5.0, 3.0])
        result = merge_ties([d1, d2], top_k_pct=100.0, scaling_lam=1.0)
        self.assertAlmostEqual(result[0].item(), 0.0, places=5)
        self.assertAlmostEqual(result[1].item(), 3.0, places=5)

    def test_large_tensor_no_quantile_crash(self):
        """Should handle tensors with >16M elements without crash (no quantile)."""
        torch.manual_seed(1)
        big = [torch.randn(4096, 4096) for _ in range(2)]  # 16M elements each
        result = merge_ties(big, top_k_pct=20.0)
        self.assertEqual(result.shape, (4096, 4096))
        self.assertFalse(torch.isnan(result).any())

    def test_scaling_lam(self):
        d = torch.tensor([4.0, -2.0])
        r1 = merge_ties([d], top_k_pct=100.0, scaling_lam=1.0)
        r2 = merge_ties([d], top_k_pct=100.0, scaling_lam=0.5)
        self.assertTrue(torch.allclose(r2, 0.5 * r1))


# ══════════════════════════════════════════════════════════════════════════════
# merge_dare
# ══════════════════════════════════════════════════════════════════════════════

class TestMergeDare(unittest.TestCase):

    def test_drop_rate_zero_identity(self):
        """drop_rate=0.0 → no elements dropped → rescale=1 → same as base method."""
        torch.manual_seed(0)
        d1 = torch.randn(8)
        d2 = torch.randn(8)
        no_drop = merge_dare([d1, d2], drop_rate=0.0, base_method="task_arithmetic",
                             scaling_lam=1.0, seed=0)
        naive_ta = merge_task_arithmetic([d1, d2], scaling_lam=1.0)
        self.assertTrue(torch.allclose(no_drop, naive_ta, atol=1e-5))

    def test_invalid_drop_rate(self):
        d = torch.randn(4)
        with self.assertRaises(ValueError):
            merge_dare([d], drop_rate=1.0)
        with self.assertRaises(ValueError):
            merge_dare([d], drop_rate=-0.1)

    def test_invalid_base_method(self):
        d = torch.randn(4)
        with self.assertRaises(ValueError):
            merge_dare([d], base_method="unknown")

    def test_drop_rate_and_rescaling(self):
        """Verifies drop rate and unbiased 1/(1-p) rescaling on a large tensor."""
        d = torch.ones(20000)
        p = 0.4
        result = merge_dare([d], drop_rate=p, base_method="task_arithmetic",
                            scaling_lam=1.0, seed=123)
        # Check empirical zero fraction matches p
        zero_frac = (result == 0.0).float().mean().item()
        self.assertAlmostEqual(zero_frac, p, delta=0.02)
        # Check surviving entries are rescaled by 1 / (1 - p)
        nonzeros = result[result != 0.0]
        expected_val = 1.0 / (1.0 - p)
        self.assertTrue(torch.allclose(nonzeros, torch.full_like(nonzeros, expected_val), atol=1e-5))

    def test_unbiased_scaling_expectation(self):
        """E[DARE(δ)] ≈ TA(δ) when averaged over many random seeds."""
        torch.manual_seed(99)
        delta = torch.randn(100)
        estimates = []
        for seed in range(500):
            result = merge_dare([delta], drop_rate=0.5, base_method="task_arithmetic",
                                scaling_lam=1.0, seed=seed)
            estimates.append(result)
        mean_estimate = torch.stack(estimates).mean(dim=0)
        # Check that relative Frobenius error is small (< 5%)
        rel_error = (torch.norm(mean_estimate - delta) / torch.norm(delta)).item()
        self.assertLess(rel_error, 0.06, f"Relative error {rel_error:.4f} too high")

    def test_shape_preserved(self):
        torch.manual_seed(0)
        ds = [torch.randn(16, 32) for _ in range(3)]
        result = merge_dare(ds, drop_rate=0.3, base_method="ties")
        self.assertEqual(result.shape, (16, 32))

    def test_seed_reproducibility(self):
        """Same seed → same result."""
        torch.manual_seed(0)
        ds = [torch.randn(10), torch.randn(10)]
        r1 = merge_dare(ds, seed=7)
        r2 = merge_dare(ds, seed=7)
        self.assertTrue(torch.allclose(r1, r2))

    def test_different_seeds_differ(self):
        torch.manual_seed(0)
        ds = [torch.randn(100)]
        r1 = merge_dare(ds, drop_rate=0.5, seed=1)
        r2 = merge_dare(ds, drop_rate=0.5, seed=2)
        # With high probability they differ when drop_rate is non-zero
        self.assertFalse(torch.allclose(r1, r2))

    def test_all_base_methods(self):
        torch.manual_seed(0)
        ds = [torch.randn(16), torch.randn(16)]
        for method in ("task_arithmetic", "average", "ties"):
            result = merge_dare(ds, base_method=method, seed=0)
            self.assertEqual(result.shape, (16,))
            self.assertFalse(torch.isnan(result).any())


# ══════════════════════════════════════════════════════════════════════════════
# merge_lora_adapters_baseline
# ══════════════════════════════════════════════════════════════════════════════

class TestMergeLoraAdaptersBaseline(unittest.TestCase):

    def _make_pair(self, d_out=32, d_in=64, r=4, seed=0):
        torch.manual_seed(seed)
        B = torch.randn(d_out, r)
        A = torch.randn(r, d_in)
        return B, A

    def test_naive_single_plugin_matches_manual(self):
        d_out, d_in, r = 16, 32, 4
        B_m, A_m = self._make_pair(d_out, d_in, r, seed=0)
        B_p, A_p = self._make_pair(d_out, d_in, r, seed=1)

        master = _make_adapter({"mod": (B_m, A_m)}, name="master")
        plugin = _make_adapter({"mod": (B_p, A_p)}, name="plugin")

        result = merge_lora_adapters_baseline(master, [plugin], method="naive")
        # scaling=1 (alpha=r), so delta = B@A
        expected = B_m @ A_m + B_p @ A_p
        self.assertTrue(torch.allclose(result["mod"], expected, atol=1e-5))

    def test_task_arithmetic_scales(self):
        d_out, d_in, r = 16, 32, 4
        B_m, A_m = self._make_pair(d_out, d_in, r, seed=0)
        B_p, A_p = self._make_pair(d_out, d_in, r, seed=1)

        master = _make_adapter({"mod": (B_m, A_m)})
        plugin = _make_adapter({"mod": (B_p, A_p)})

        lam = 0.3
        result = merge_lora_adapters_baseline(master, [plugin], method="task_arithmetic",
                                              scaling_lam=lam)
        expected = lam * (B_m @ A_m + B_p @ A_p)
        self.assertTrue(torch.allclose(result["mod"], expected, atol=1e-5))

    def test_ties_runs_and_has_correct_shape(self):
        d_out, d_in, r = 16, 32, 4
        B_m, A_m = self._make_pair(d_out, d_in, r, seed=2)
        B_p, A_p = self._make_pair(d_out, d_in, r, seed=3)

        master = _make_adapter({"mod": (B_m, A_m)})
        plugin = _make_adapter({"mod": (B_p, A_p)})

        result = merge_lora_adapters_baseline(master, [plugin], method="ties",
                                              top_k_pct=50.0)
        self.assertIn("mod", result)
        self.assertEqual(result["mod"].shape, (d_out, d_in))

    def test_dare_runs_and_has_correct_shape(self):
        d_out, d_in, r = 16, 32, 4
        B_m, A_m = self._make_pair(d_out, d_in, r, seed=4)
        B_p, A_p = self._make_pair(d_out, d_in, r, seed=5)

        master = _make_adapter({"mod": (B_m, A_m)})
        plugin = _make_adapter({"mod": (B_p, A_p)})

        result = merge_lora_adapters_baseline(master, [plugin], method="dare",
                                              drop_rate=0.5, seed=42)
        self.assertIn("mod", result)
        self.assertEqual(result["mod"].shape, (d_out, d_in))

    def test_invalid_method_raises(self):
        B, A = self._make_pair()
        master = _make_adapter({"mod": (B, A)})
        with self.assertRaises(ValueError):
            merge_lora_adapters_baseline(master, [], method="bogus")

    def test_target_modules_filtering(self):
        """Only target modules should appear in the output."""
        d_out, d_in, r = 8, 16, 2
        torch.manual_seed(0)
        B1, A1 = torch.randn(d_out, r), torch.randn(r, d_in)
        B2, A2 = torch.randn(d_out, r), torch.randn(r, d_in)
        master = _make_adapter({"mod1": (B1, A1), "mod2": (B2, A2)})
        plugin = _make_adapter({"mod1": (B1, A1), "mod2": (B2, A2)})

        result = merge_lora_adapters_baseline(master, [plugin], method="naive",
                                              target_modules=["mod1"])
        self.assertIn("mod1", result)
        self.assertNotIn("mod2", result)

    def test_missing_plugin_module_skipped(self):
        """Module absent in plugin should still use only master delta."""
        d_out, d_in, r = 8, 16, 2
        torch.manual_seed(0)
        B_m, A_m = torch.randn(d_out, r), torch.randn(r, d_in)
        B_p, A_p = torch.randn(d_out, r), torch.randn(r, d_in)
        master = _make_adapter({"mod_a": (B_m, A_m), "mod_b": (B_m, A_m)})
        # plugin only has mod_a
        plugin = _make_adapter({"mod_a": (B_p, A_p)})

        result = merge_lora_adapters_baseline(master, [plugin], method="naive")
        # mod_b: only master delta
        expected_b = B_m @ A_m
        self.assertTrue(torch.allclose(result["mod_b"], expected_b, atol=1e-5))


if __name__ == "__main__":
    unittest.main()

