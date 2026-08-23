"""High-level merger pipeline implementing Two-Sided Asymmetric Null-Space Projection."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from tqdm import tqdm

from asym_nsp.core import (
    compute_orthogonality_metrics,
    project_two_sided_nsp,
)
from asym_nsp.lora_loader import (
    LoRAAdapter,
    LoRALayerWeights,
    load_lora_adapter,
)


@dataclass
class MergeConfig:
    """Configuration settings for Asymmetric NSP LoRA merge."""
    master_adapter_path: str
    plugin_adapter_paths: List[str] = field(default_factory=list)
    output_dir: str = "./merged_model"
    energy_threshold: Optional[float] = 0.99
    rank_k: Optional[int] = None
    project_left: bool = True
    project_right: bool = True
    plugin_weights: Optional[List[float]] = None
    target_modules: Optional[List[str]] = None
    device: str = "cpu"
    dtype: str = "float32"

    @classmethod
    def from_dict(cls, data: dict) -> "MergeConfig":
        plugins = data.get("plugin_adapter_paths") or data.get("plugins") or []
        if isinstance(plugins, str):
            plugins = [plugins]
        return cls(
            master_adapter_path=data["master_adapter_path"],
            plugin_adapter_paths=plugins,
            output_dir=data.get("output_dir", "./merged_model"),
            energy_threshold=data.get("energy_threshold", 0.99),
            rank_k=data.get("rank_k"),
            project_left=data.get("project_left", True),
            project_right=data.get("project_right", True),
            plugin_weights=data.get("plugin_weights"),
            target_modules=data.get("target_modules"),
            device=data.get("device", "cpu"),
            dtype=data.get("dtype", "float32"),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "MergeConfig":
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class AsymmetricNSPMerger:
    """Orchestrates Two-Sided Asymmetric Null-Space Projection merging across adapters."""

    def __init__(self, config: MergeConfig):
        self.config = config
        self.torch_dtype = getattr(torch, config.dtype, torch.float32)

    def merge_adapters(
        self,
        master: Optional[LoRAAdapter] = None,
        plugins: Optional[List[LoRAAdapter]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Merges plugin LoRA adapters into the null-space of the master adapter.

        Returns:
            Dictionary mapping module_key to merged full-rank delta_W tensor.
        """
        # Load adapters if not provided as instances
        if master is None:
            master = load_lora_adapter(
                self.config.master_adapter_path,
                name="master",
                device=self.config.device,
                dtype=self.torch_dtype,
            )

        if plugins is None:
            plugins = [
                load_lora_adapter(
                    p_path,
                    name=f"plugin_{i}",
                    device=self.config.device,
                    dtype=self.torch_dtype,
                )
                for i, p_path in enumerate(self.config.plugin_adapter_paths)
            ]

        plugin_weights = self.config.plugin_weights or [1.0] * len(plugins)
        merged_deltas: Dict[str, torch.Tensor] = {}
        layer_metrics: Dict[str, list] = {}

        master_modules = master.module_keys()

        for mod_key in master_modules:
            # Check target module filter if specified
            if self.config.target_modules:
                if not any(target in mod_key for target in self.config.target_modules):
                    continue

            master_layer = master.get_layer(mod_key)
            if master_layer is None:
                continue

            # Start with exact master delta weight (100% preservation)
            delta_w_accum = master_layer.compute_delta_w()

            for p_idx, plugin in enumerate(plugins):
                plugin_layer = plugin.get_layer(mod_key)
                if plugin_layer is None:
                    continue

                p_weight = plugin_weights[p_idx]

                # Project plugin into null-space of master
                b_proj, a_proj = project_two_sided_nsp(
                    B_plugin=plugin_layer.weight_b,
                    A_plugin=plugin_layer.weight_a,
                    B_master=master_layer.weight_b,
                    A_master=master_layer.weight_a,
                    rank_k=self.config.rank_k,
                    energy_threshold=self.config.energy_threshold,
                    project_left=self.config.project_left,
                    project_right=self.config.project_right,
                )

                # Record verification metrics
                metrics = compute_orthogonality_metrics(
                    B_plugin_proj=b_proj,
                    A_plugin_proj=a_proj,
                    B_master=master_layer.weight_b,
                    A_master=master_layer.weight_a,
                )
                metrics["plugin_name"] = plugin.name
                metrics["module"] = mod_key
                layer_metrics.setdefault(mod_key, []).append(metrics)

                # Add projected plugin delta W
                delta_w_plugin = plugin_layer.scaling * torch.matmul(b_proj, a_proj)
                delta_w_accum = delta_w_accum + p_weight * delta_w_plugin

            merged_deltas[mod_key] = delta_w_accum

        return merged_deltas

    def apply_and_save(
        self,
        base_model_path: str,
        output_dir: Union[str, Path],
        merged_deltas: Dict[str, torch.Tensor],
        save_tokenizer: bool = True,
    ):
        """Applies merged delta weights onto base model and saves merged weights."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from safetensors.torch import save_file

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"Loading base model from {base_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.config.device != "cpu" else None,
        )

        state_dict = model.state_dict()
        for mod_key, delta_w in tqdm(merged_deltas.items(), desc="Applying merged delta weights"):
            # Match parameter key in base model
            target_key = f"{mod_key}.weight"
            if target_key in state_dict:
                state_dict[target_key] += delta_w.to(
                    device=state_dict[target_key].device,
                    dtype=state_dict[target_key].dtype,
                )
            elif f"model.{mod_key}.weight" in state_dict:
                state_dict[f"model.{mod_key}.weight"] += delta_w.to(
                    device=state_dict[f"model.{mod_key}.weight"].device,
                    dtype=state_dict[f"model.{mod_key}.weight"].dtype,
                )

        print(f"Saving merged model to {out_path}...")
        model.save_pretrained(out_path)

        if save_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            tokenizer.save_pretrained(out_path)

        print("Merge successfully completed and saved!")
