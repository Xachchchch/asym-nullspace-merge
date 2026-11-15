"""Unit tests for dimension handling, different ranks, merger scaling, and PEFT export."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asym_nsp.core import (
    compute_subspace_basis,
    project_left_null_space,
    project_right_null_space,
    project_two_sided_nsp,
)
from asym_nsp.lora_loader import (
    LoRAAdapter,
    LoRALayerWeights,
    _normalize_module_key,
    load_lora_adapter,
)
from asym_nsp.merger import AsymmetricNSPMerger, MergeConfig


class TestMergeShapes(unittest.TestCase):

    def test_various_shapes_and_ranks(self):
        """Checks that projection maintains input and output shapes regardless of rank."""
        test_cases = [
            (4096, 4096, 16, 16),
            (4096, 11008, 32, 8),
            (11008, 4096, 8, 64),
            (128, 256, 4, 4),
        ]
        for d_out, d_in, r_m, r_p in test_cases:
            torch.manual_seed(0)
            B_m = torch.randn(d_out, r_m)
            A_m = torch.randn(r_m, d_in)
            B_p = torch.randn(d_out, r_p)
            A_p = torch.randn(r_p, d_in)

            B_p_proj, A_p_proj = project_two_sided_nsp(
                B_plugin=B_p,
                A_plugin=A_p,
                B_master=B_m,
                A_master=A_m,
            )

            self.assertEqual(B_p_proj.shape, (d_out, r_p))
            self.assertEqual(A_p_proj.shape, (r_p, d_in))

            delta_w_m = torch.matmul(B_m, A_m)
            delta_w_p = torch.matmul(B_p_proj, A_p_proj)

            self.assertEqual(delta_w_m.shape, (d_out, d_in))
            self.assertEqual(delta_w_p.shape, (d_out, d_in))

    def test_asymmetric_merger_synthetic_adapters(self):
        """Checks end-to-end AsymmetricNSPMerger with synthetic LoRAAdapter instances."""
        d_out, d_in = 128, 128

        master_layer = LoRALayerWeights(
            module_key="model.layers.0.self_attn.q_proj",
            weight_a=torch.randn(8, d_in),
            weight_b=torch.randn(d_out, 8),
            r=8,
            lora_alpha=16.0,
        )
        master = LoRAAdapter(
            name="master_synthetic",
            config={"r": 8, "lora_alpha": 16.0},
            layers={"model.layers.0.self_attn.q_proj": master_layer},
        )

        plugin_layer = LoRALayerWeights(
            module_key="model.layers.0.self_attn.q_proj",
            weight_a=torch.randn(16, d_in),
            weight_b=torch.randn(d_out, 16),
            r=16,
            lora_alpha=32.0,
        )
        plugin = LoRAAdapter(
            name="plugin_synthetic",
            config={"r": 16, "lora_alpha": 32.0},
            layers={"model.layers.0.self_attn.q_proj": plugin_layer},
        )

        config = MergeConfig(
            master_adapter_path="synthetic_master",
            plugin_adapter_paths=["synthetic_plugin"],
        )
        merger = AsymmetricNSPMerger(config)

        merged_deltas = merger.merge_adapters(master=master, plugins=[plugin])

        self.assertIn("model.layers.0.self_attn.q_proj", merged_deltas)
        delta_w = merged_deltas["model.layers.0.self_attn.q_proj"]
        self.assertEqual(delta_w.shape, (d_out, d_in))
        self.assertFalse(torch.isnan(delta_w).any())

    def test_export_as_peft_adapter(self):
        """Verifies exact numerical equivalence ((alpha / effective_r) * (B_peft @ A_peft) == s_m B_m A_m + s_p B'_p A'_p) and config."""
        torch.manual_seed(42)
        d_out, d_in = 64, 64
        r_m, r_p = 8, 8
        alpha_m, alpha_p = 16.0, 16.0

        master_layer = LoRALayerWeights(
            module_key="model.layers.0.self_attn.q_proj",
            weight_a=torch.randn(r_m, d_in, dtype=torch.float64),
            weight_b=torch.randn(d_out, r_m, dtype=torch.float64),
            r=r_m,
            lora_alpha=alpha_m,
        )
        master = LoRAAdapter(
            name="master_synthetic",
            config={"r": r_m, "lora_alpha": alpha_m},
            layers={"model.layers.0.self_attn.q_proj": master_layer},
        )

        plugin_layer = LoRALayerWeights(
            module_key="model.layers.0.self_attn.q_proj",
            weight_a=torch.randn(r_p, d_in, dtype=torch.float64),
            weight_b=torch.randn(d_out, r_p, dtype=torch.float64),
            r=r_p,
            lora_alpha=alpha_p,
        )
        plugin = LoRAAdapter(
            name="plugin_synthetic",
            config={"r": r_p, "lora_alpha": alpha_p},
            layers={"model.layers.0.self_attn.q_proj": plugin_layer},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target_alpha = 32.0
            config = MergeConfig(
                master_adapter_path="synthetic_master",
                plugin_adapter_paths=["synthetic_plugin"],
                output_dir=tmpdir,
                lora_alpha=target_alpha,
                dtype="float64",
            )
            merger = AsymmetricNSPMerger(config)
            merger.export_as_peft_adapter(
                output_dir=tmpdir,
                master=master,
                plugins=[plugin],
            )

            cfg_path = Path(tmpdir) / "adapter_config.json"
            safetensors_path = Path(tmpdir) / "adapter_model.safetensors"
            bin_path = Path(tmpdir) / "adapter_model.bin"

            self.assertTrue(cfg_path.exists())
            self.assertTrue(safetensors_path.exists() or bin_path.exists())

            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            self.assertEqual(cfg["peft_type"], "LORA")
            self.assertEqual(cfg["lora_alpha"], target_alpha)
            self.assertEqual(cfg["r"], r_m + r_p)
            self.assertIn("q_proj", cfg["target_modules"])

            if safetensors_path.exists():
                from safetensors.torch import load_file
                loaded_sd = load_file(str(safetensors_path))
            else:
                loaded_sd = torch.load(str(bin_path), map_location="cpu")

            key_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            key_b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
            self.assertIn(key_a, loaded_sd)
            self.assertIn(key_b, loaded_sd)

            A_peft = loaded_sd[key_a].to(dtype=torch.float64)
            B_peft = loaded_sd[key_b].to(dtype=torch.float64)

            effective_r = cfg["r"]
            peft_delta = (target_alpha / effective_r) * torch.matmul(B_peft, A_peft)

            # Ground truth reconstructed delta
            b_p_proj, a_p_proj = project_two_sided_nsp(
                B_plugin=plugin_layer.weight_b,
                A_plugin=plugin_layer.weight_a,
                B_master=master_layer.weight_b,
                A_master=master_layer.weight_a,
                energy_threshold=1.0,
            )
            s_m = master_layer.scaling
            s_p = plugin_layer.scaling
            expected_delta = s_m * torch.matmul(master_layer.weight_b, master_layer.weight_a) + \
                             s_p * torch.matmul(b_p_proj, a_p_proj)

            diff = torch.norm(peft_delta - expected_delta, p="fro").item()
            self.assertLess(diff, 1e-5, f"PEFT export numerical difference {diff} exceeds 1e-5")

    def test_chained_multi_plugin_merging(self):
        """Verifies Chained N = 3 plugins: Plugin 3 has < 1e-5 leakage against Master, Plugin 1, and Plugin 2."""
        torch.manual_seed(42)
        d_out, d_in = 64, 64
        r = 4

        m_layer = LoRALayerWeights("layer", torch.randn(r, d_in), torch.randn(d_out, r), r=r)
        p1_layer = LoRALayerWeights("layer", torch.randn(r, d_in), torch.randn(d_out, r), r=r)
        p2_layer = LoRALayerWeights("layer", torch.randn(r, d_in), torch.randn(d_out, r), r=r)
        p3_layer = LoRALayerWeights("layer", torch.randn(r, d_in), torch.randn(d_out, r), r=r)

        master = LoRAAdapter("m", {}, {"layer": m_layer})
        p1 = LoRAAdapter("p1", {}, {"layer": p1_layer})
        p2 = LoRAAdapter("p2", {}, {"layer": p2_layer})
        p3 = LoRAAdapter("p3", {}, {"layer": p3_layer})

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MergeConfig(
                master_adapter_path="m",
                plugin_adapter_paths=["p1", "p2", "p3"],
                output_dir=tmpdir,
                energy_threshold=1.0,
            )
            merger = AsymmetricNSPMerger(config)
            merger.export_as_peft_adapter(tmpdir, master=master, plugins=[p1, p2, p3])

            safetensors_path = Path(tmpdir) / "adapter_model.safetensors"
            bin_path = Path(tmpdir) / "adapter_model.bin"
            if safetensors_path.exists():
                from safetensors.torch import load_file
                sd = load_file(str(safetensors_path))
            else:
                sd = torch.load(str(bin_path), map_location="cpu")

            key_b = "base_model.model.layer.lora_B.weight"
            key_a = "base_model.model.layer.lora_A.weight"
            B_all = sd[key_b].double()
            A_all = sd[key_a].double()

            # Columns/rows correspond to: Master (0:4), P1 (4:8), P2 (8:12), P3 (12:16)
            B_m, A_m = B_all[:, 0:4], A_all[0:4, :]
            B_p1, A_p1 = B_all[:, 4:8], A_all[4:8, :]
            B_p2, A_p2 = B_all[:, 8:12], A_all[8:12, :]
            B_p3, A_p3 = B_all[:, 12:16], A_all[12:16, :]

            # Test Plugin 3 leakage against Master, Plugin 1, and Plugin 2
            for name, B_prior, A_prior in [("Master", B_m, A_m), ("Plugin 1", B_p1, A_p1), ("Plugin 2", B_p2, A_p2)]:
                U_prior, _ = compute_subspace_basis(B_prior, basis_side="left")
                leakage_b = torch.norm(torch.matmul(U_prior.T, B_p3), p="fro") / (torch.norm(B_p3, p="fro") + 1e-12)
                self.assertLess(leakage_b.item(), 1e-5, f"P3 B-leakage against {name} is {leakage_b.item()}")

                V_prior, _ = compute_subspace_basis(A_prior, basis_side="right")
                leakage_a = torch.norm(torch.matmul(A_p3, V_prior), p="fro") / (torch.norm(A_p3, p="fro") + 1e-12)
                self.assertLess(leakage_a.item(), 1e-5, f"P3 A-leakage against {name} is {leakage_a.item()}")

    def test_negative_plugin_weights(self):
        """Verifies plugin_weights=[-0.5] produces no NaNs and reconstructs exact delta."""
        torch.manual_seed(42)
        d_out, d_in = 32, 32
        r = 4

        m_layer = LoRALayerWeights("layer", torch.randn(r, d_in, dtype=torch.float64), torch.randn(d_out, r, dtype=torch.float64), r=r, lora_alpha=8.0)
        p_layer = LoRALayerWeights("layer", torch.randn(r, d_in, dtype=torch.float64), torch.randn(d_out, r, dtype=torch.float64), r=r, lora_alpha=8.0)

        master = LoRAAdapter("m", {"r": r, "lora_alpha": 8.0}, {"layer": m_layer})
        plugin = LoRAAdapter("p", {"r": r, "lora_alpha": 8.0}, {"layer": p_layer})

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MergeConfig(
                master_adapter_path="m",
                plugin_adapter_paths=["p"],
                plugin_weights=[-0.5],
                lora_alpha=16.0,
                dtype="float64",
                output_dir=tmpdir,
            )
            merger = AsymmetricNSPMerger(config)
            merger.export_as_peft_adapter(tmpdir, master=master, plugins=[plugin])

            safetensors_path = Path(tmpdir) / "adapter_model.safetensors"
            bin_path = Path(tmpdir) / "adapter_model.bin"
            if safetensors_path.exists():
                from safetensors.torch import load_file
                sd = load_file(str(safetensors_path))
            else:
                sd = torch.load(str(bin_path), map_location="cpu")

            A_peft = sd["base_model.model.layer.lora_A.weight"].to(dtype=torch.float64)
            B_peft = sd["base_model.model.layer.lora_B.weight"].to(dtype=torch.float64)

            self.assertFalse(torch.isnan(A_peft).any(), "Found NaNs in A_peft")
            self.assertFalse(torch.isnan(B_peft).any(), "Found NaNs in B_peft")

            effective_r = 8
            peft_delta = (16.0 / effective_r) * torch.matmul(B_peft, A_peft)

            b_proj, a_proj = project_two_sided_nsp(
                p_layer.weight_b, p_layer.weight_a, m_layer.weight_b, m_layer.weight_a, energy_threshold=1.0
            )
            expected_delta = m_layer.compute_delta_w() + (-0.5) * p_layer.scaling * torch.matmul(b_proj, a_proj)

            diff = torch.norm(peft_delta - expected_delta, p="fro").item()
            self.assertLess(diff, 1e-5, f"Negative weight delta diff: {diff}")

    def test_target_r_truncation(self):
        """Verifies target_r truncation keeps Master subspace 100% invariant and raises ValueError on t_r < r_master."""
        torch.manual_seed(42)
        d_out, d_in = 64, 64
        r_m, r_p = 8, 8

        m_layer = LoRALayerWeights("layer", torch.randn(r_m, d_in), torch.randn(d_out, r_m), r=r_m, lora_alpha=16.0)
        p_layer = LoRALayerWeights("layer", torch.randn(r_p, d_in), torch.randn(d_out, r_p), r=r_p, lora_alpha=16.0)

        master = LoRAAdapter("m", {"r": r_m, "lora_alpha": 16.0}, {"layer": m_layer})
        plugin = LoRAAdapter("p", {"r": r_p, "lora_alpha": 16.0}, {"layer": p_layer})

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MergeConfig(master_adapter_path="m", plugin_adapter_paths=["p"], output_dir=tmpdir)
            merger = AsymmetricNSPMerger(config)

            # Must raise ValueError when target_r < r_master
            with self.assertRaises(ValueError):
                merger.export_as_peft_adapter(tmpdir, master=master, plugins=[plugin], target_r=4)

            # Truncate to target_r = 12 (where r_master = 8, plugin block compressed from 8 to 4)
            merger.export_as_peft_adapter(tmpdir, master=master, plugins=[plugin], target_r=12)

            safetensors_path = Path(tmpdir) / "adapter_model.safetensors"
            bin_path = Path(tmpdir) / "adapter_model.bin"
            if safetensors_path.exists():
                from safetensors.torch import load_file
                sd = load_file(str(safetensors_path))
            else:
                sd = torch.load(str(bin_path), map_location="cpu")

            A_peft = sd["base_model.model.layer.lora_A.weight"].double()
            B_peft = sd["base_model.model.layer.lora_B.weight"].double()

            self.assertEqual(B_peft.shape[1], 12)
            self.assertEqual(A_peft.shape[0], 12)

            # Master block unscaling:
            effective_r = 12
            peft_scaling = config.lora_alpha / effective_r
            scale_factor = (1.0 / peft_scaling) ** 0.5
            s_m_mag = abs(m_layer.scaling) ** 0.5

            B_master_recovered = (B_peft[:, :r_m] / scale_factor) / s_m_mag
            A_master_recovered = (A_peft[:r_m, :] / scale_factor) / s_m_mag

            diff_b = torch.norm(B_master_recovered - m_layer.weight_b.double(), p="fro").item()
            diff_a = torch.norm(A_master_recovered - m_layer.weight_a.double(), p="fro").item()
            self.assertLess(diff_b, 1e-5, f"Master B shifted by {diff_b}")
            self.assertLess(diff_a, 1e-5, f"Master A shifted by {diff_a}")

    def test_custom_adapter_name_loading(self):
        """Verifies _normalize_module_key and load_lora_adapter with custom adapter names like .math_adapter."""
        custom_key_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.math_adapter.weight"
        custom_key_b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.math_adapter.weight"

        clean_a = _normalize_module_key(custom_key_a)
        clean_b = _normalize_module_key(custom_key_b)
        self.assertEqual(clean_a, "model.layers.0.self_attn.q_proj")
        self.assertEqual(clean_b, "model.layers.0.self_attn.q_proj")

        with tempfile.TemporaryDirectory() as tmpdir:
            from safetensors.torch import save_file
            sd = {
                custom_key_a: torch.randn(4, 32),
                custom_key_b: torch.randn(32, 4),
            }
            save_file(sd, str(Path(tmpdir) / "adapter_model.safetensors"))
            cfg = {"r": 4, "lora_alpha": 8.0, "fan_in_fan_out": False}
            with open(Path(tmpdir) / "adapter_config.json", "w") as f:
                json.dump(cfg, f)

            adapter = load_lora_adapter(tmpdir)
            self.assertIn("model.layers.0.self_attn.q_proj", adapter.layers)
            layer = adapter.get_layer("model.layers.0.self_attn.q_proj")
            self.assertEqual(layer.r, 4)
            self.assertEqual(layer.weight_a.shape, (4, 32))
            self.assertEqual(layer.weight_b.shape, (32, 4))

    def test_real_transformers_peft_end_to_end(self):
        """Tests end-to-end PeftModel loading & forward pass with tiny LLaMA for both homogeneous and heterogeneous adapters."""
        from transformers import LlamaConfig, LlamaForCausalLM
        from peft import PeftModel

        cfg = LlamaConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
        )
        base_model = LlamaForCausalLM(cfg)

        d_out, d_in = 32, 32
        m_q = LoRALayerWeights("model.layers.0.self_attn.q_proj", torch.randn(4, d_in), torch.randn(d_out, 4), r=4, lora_alpha=8.0)
        p_q = LoRALayerWeights("model.layers.0.self_attn.q_proj", torch.randn(4, d_in), torch.randn(d_out, 4), r=4, lora_alpha=8.0)

        # 1. Homogeneous test: both adapt q_proj
        master_homo = LoRAAdapter("m_homo", {"r": 4, "lora_alpha": 8.0}, {"model.layers.0.self_attn.q_proj": m_q})
        plugin_homo = LoRAAdapter("p_homo", {"r": 4, "lora_alpha": 8.0}, {"model.layers.0.self_attn.q_proj": p_q})

        with tempfile.TemporaryDirectory() as tmpdir:
            merger = AsymmetricNSPMerger(MergeConfig(master_adapter_path="m", plugin_adapter_paths=["p"]))
            merger.export_as_peft_adapter(tmpdir, master=master_homo, plugins=[plugin_homo])

            peft_m = PeftModel.from_pretrained(base_model, tmpdir)
            x = torch.randint(0, 100, (1, 8))
            out = peft_m(x)
            self.assertEqual(out.logits.shape, (1, 8, 100))
            self.assertFalse(torch.isnan(out.logits).any())

        # 2. Heterogeneous test: master adapts q_proj + k_proj, plugin only adapts q_proj
        m_k = LoRALayerWeights("model.layers.0.self_attn.k_proj", torch.randn(4, d_in), torch.randn(d_out, 4), r=4, lora_alpha=8.0)
        master_hetero = LoRAAdapter("m_hetero", {"r": 4, "lora_alpha": 8.0}, {
            "model.layers.0.self_attn.q_proj": m_q,
            "model.layers.0.self_attn.k_proj": m_k,
        })
        plugin_hetero = LoRAAdapter("p_hetero", {"r": 4, "lora_alpha": 8.0}, {
            "model.layers.0.self_attn.q_proj": p_q,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            merger = AsymmetricNSPMerger(MergeConfig(master_adapter_path="m", plugin_adapter_paths=["p"]))
            merger.export_as_peft_adapter(tmpdir, master=master_hetero, plugins=[plugin_hetero])

            with open(Path(tmpdir) / "adapter_config.json") as f:
                adapter_cfg = json.load(f)
            self.assertEqual(adapter_cfg["r"], 8)
            self.assertIn("rank_pattern", adapter_cfg)
            self.assertEqual(adapter_cfg["rank_pattern"]["model.layers.0.self_attn.q_proj"], 8)
            self.assertEqual(adapter_cfg["rank_pattern"]["model.layers.0.self_attn.k_proj"], 4)

            peft_hetero = PeftModel.from_pretrained(base_model, tmpdir)
            x = torch.randint(0, 100, (1, 8))
            out_hetero = peft_hetero(x)
            self.assertEqual(out_hetero.logits.shape, (1, 8, 100))
            self.assertFalse(torch.isnan(out_hetero.logits).any())


if __name__ == "__main__":
    unittest.main()
