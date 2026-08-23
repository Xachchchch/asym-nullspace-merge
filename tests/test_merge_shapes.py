"""Unit tests for dimension handling, different ranks, and merger scaling."""

import pytest
import torch

from asym_nsp.core import project_two_sided_nsp
from asym_nsp.lora_loader import LoRAAdapter, LoRALayerWeights
from asym_nsp.merger import AsymmetricNSPMerger, MergeConfig


@pytest.mark.parametrize(
    "d_out,d_in,r_m,r_p",
    [
        (4096, 4096, 16, 16),
        (4096, 11008, 32, 8),
        (11008, 4096, 8, 64),
        (128, 256, 4, 4),
    ],
)
def test_various_shapes_and_ranks(d_out, d_in, r_m, r_p):
    """Checks that projection maintains input and output shapes regardless of rank."""
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

    assert B_p_proj.shape == (d_out, r_p)
    assert A_p_proj.shape == (r_p, d_in)

    delta_w_m = torch.matmul(B_m, A_m)
    delta_w_p = torch.matmul(B_p_proj, A_p_proj)

    assert delta_w_m.shape == (d_out, d_in)
    assert delta_w_p.shape == (d_out, d_in)


def test_asymmetric_merger_synthetic_adapters():
    """Checks end-to-end AsymmetricNSPMerger with synthetic LoRAAdapter instances."""
    d_out, d_in = 128, 128
    
    # Create master adapter
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

    # Create plugin adapter
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

    assert "model.layers.0.self_attn.q_proj" in merged_deltas
    delta_w = merged_deltas["model.layers.0.self_attn.q_proj"]
    assert delta_w.shape == (d_out, d_in)
    assert not torch.isnan(delta_w).any()
