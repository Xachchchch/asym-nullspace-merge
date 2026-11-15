"""High-level merger pipeline implementing Two-Sided Asymmetric Null-Space Projection."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from tqdm import tqdm

from asym_nsp.core import (
    build_cumulative_subspace_basis,
    compute_orthogonality_metrics,
    project_left_null_space,
    project_right_null_space,
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
    energy_threshold: Optional[float] = 1.0
    rank_k: Optional[int] = None
    project_left: bool = True
    project_right: bool = True
    plugin_weights: Optional[List[float]] = None
    target_modules: Optional[List[str]] = None
    export_adapter: bool = False
    target_r: Optional[int] = None
    lora_alpha: float = 32.0
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
            energy_threshold=data["energy_threshold"] if "energy_threshold" in data else 1.0,
            rank_k=data.get("rank_k"),
            project_left=data.get("project_left", True),
            project_right=data.get("project_right", True),
            plugin_weights=data.get("plugin_weights"),
            target_modules=data.get("target_modules"),
            export_adapter=data.get("export_adapter", False),
            target_r=data.get("target_r"),
            lora_alpha=data.get("lora_alpha", 32.0),
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
        return_metrics: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, list]]]:
        """Merges plugin LoRA adapters into the null-space of the master adapter.

        Supports multi-adapter chaining (N > 1) where each subsequent plugin is
        projected into the cumulative null-space of Master and all prior plugins.

        Args:
            master: Optional preloaded master adapter instance.
            plugins: Optional list of preloaded plugin adapter instances.
            return_metrics: Whether to compute and return verification metrics.

        Returns:
            Dictionary mapping module_key to merged full-rank delta_W tensor,
            or tuple of (merged_deltas, layer_metrics) if return_metrics is True.
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

            # Maintain protected basis lists for multi-adapter chaining
            protected_B_list = [master_layer.weight_b]
            protected_A_list = [master_layer.weight_a]

            for p_idx, plugin in enumerate(plugins):
                plugin_layer = plugin.get_layer(mod_key)
                if plugin_layer is None:
                    continue

                p_weight = plugin_weights[p_idx]
                if p_weight == 0:
                    continue

                # Project plugin into cumulative null-space of master and all prior plugins
                if self.config.project_left:
                    U_cum = build_cumulative_subspace_basis(
                        protected_B_list,
                        rank=self.config.rank_k,
                        energy_threshold=self.config.energy_threshold,
                        basis_side="left",
                    )
                    b_proj = project_left_null_space(plugin_layer.weight_b, U_cum)
                else:
                    b_proj = plugin_layer.weight_b.clone()

                if self.config.project_right:
                    V_cum = build_cumulative_subspace_basis(
                        protected_A_list,
                        rank=self.config.rank_k,
                        energy_threshold=self.config.energy_threshold,
                        basis_side="right",
                    )
                    a_proj = project_right_null_space(plugin_layer.weight_a, V_cum)
                else:
                    a_proj = plugin_layer.weight_a.clone()

                protected_B_list.append(b_proj)
                protected_A_list.append(a_proj)

                # Add projected plugin delta W
                prod = torch.matmul(b_proj, a_proj)
                if plugin_layer.fan_in_fan_out:
                    prod = prod.T
                delta_w_plugin = plugin_layer.scaling * prod
                delta_w_accum = delta_w_accum + p_weight * delta_w_plugin

                if return_metrics:
                    metrics = compute_orthogonality_metrics(
                        B_plugin_proj=b_proj,
                        A_plugin_proj=a_proj,
                        B_master=master_layer.weight_b,
                        A_master=master_layer.weight_a,
                    )
                    metrics["plugin_name"] = plugin.name
                    metrics["module"] = mod_key
                    layer_metrics.setdefault(mod_key, []).append(metrics)

            merged_deltas[mod_key] = delta_w_accum

        if return_metrics:
            return merged_deltas, layer_metrics
        return merged_deltas

    def apply_and_save(
        self,
        base_model_path: str,
        output_dir: Union[str, Path],
        merged_deltas: Dict[str, torch.Tensor],
        save_tokenizer: bool = True,
        master: Optional[LoRAAdapter] = None,
    ):
        """Applies merged delta weights onto base model and saves merged weights."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"Loading base model from {base_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.config.device != "cpu" else None,
        )

        is_fifo = master.fan_in_fan_out if master is not None else False
        if not is_fifo and self.config.master_adapter_path:
            cfg_p = Path(self.config.master_adapter_path) / "adapter_config.json"
            if cfg_p.exists():
                try:
                    with open(cfg_p, "r", encoding="utf-8") as f:
                        is_fifo = bool(json.load(f).get("fan_in_fan_out", False))
                except Exception:
                    pass

        state_dict = model.state_dict()
        for mod_key, delta_w in tqdm(merged_deltas.items(), desc="Applying merged delta weights"):
            target_key = f"{mod_key}.weight"
            target_key_alt = f"model.{mod_key}.weight"

            if target_key in state_dict:
                matched_key = target_key
            elif target_key_alt in state_dict:
                matched_key = target_key_alt
            else:
                print(f"WARNING: Module key '{mod_key}' was not found in model state_dict.")
                continue

            param_shape = state_dict[matched_key].shape
            if delta_w.shape != param_shape and delta_w.T.shape == param_shape:
                dw = delta_w.T
            elif is_fifo and delta_w.shape != param_shape:
                dw = delta_w.T
            else:
                dw = delta_w

            state_dict[matched_key] += dw.to(
                device=state_dict[matched_key].device,
                dtype=state_dict[matched_key].dtype,
            )

        print(f"Saving merged model to {out_path}...")
        model.save_pretrained(out_path)

        if save_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            tokenizer.save_pretrained(out_path)

        print("Merge successfully completed and saved!")

    def export_as_peft_adapter(
        self,
        output_dir: Union[str, Path],
        master: Optional[LoRAAdapter] = None,
        plugins: Optional[List[LoRAAdapter]] = None,
        target_r: Optional[int] = None,
        lora_alpha: Optional[float] = None,
    ):
        """Exports the merged adapter as a standalone Hugging Face PEFT adapter directory.

        Guarantees exact numerical equivalence:
            (alpha / r_eff) * (B_peft @ A_peft) == s_m B_m A_m + sum(s_p B'_p A'_p)
        Preserves Master weights without corruption during target_r truncation.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

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

        alpha = lora_alpha or self.config.lora_alpha
        if alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {alpha}")

        adapter_state_dict: Dict[str, torch.Tensor] = {}
        target_modules_set = set()
        ranks_recorded = []
        module_ranks: Dict[str, int] = {}

        master_modules = master.module_keys()
        plugin_weights = self.config.plugin_weights or [1.0] * len(plugins)

        for mod_key in tqdm(master_modules, desc="Constructing standalone PEFT adapter"):
            if self.config.target_modules:
                if not any(target in mod_key for target in self.config.target_modules):
                    continue

            master_layer = master.get_layer(mod_key)
            if master_layer is None:
                continue

            short_name = mod_key.split(".")[-1]
            target_modules_set.add(short_name)

            s_m = master_layer.scaling
            s_m_sign = 1.0 if s_m >= 0 else -1.0
            s_m_mag = abs(s_m) ** 0.5
            b_blocks = [(s_m_sign * s_m_mag) * master_layer.weight_b]
            a_blocks = [s_m_mag * master_layer.weight_a]

            protected_B_list = [master_layer.weight_b]
            protected_A_list = [master_layer.weight_a]

            for p_idx, plugin in enumerate(plugins):
                plugin_layer = plugin.get_layer(mod_key)
                if plugin_layer is None:
                    continue

                p_weight = plugin_weights[p_idx]
                if p_weight == 0:
                    continue

                if self.config.project_left:
                    U_cum = build_cumulative_subspace_basis(
                        protected_B_list,
                        rank=self.config.rank_k,
                        energy_threshold=self.config.energy_threshold,
                        basis_side="left",
                    )
                    b_proj = project_left_null_space(plugin_layer.weight_b, U_cum)
                else:
                    b_proj = plugin_layer.weight_b.clone()

                if self.config.project_right:
                    V_cum = build_cumulative_subspace_basis(
                        protected_A_list,
                        rank=self.config.rank_k,
                        energy_threshold=self.config.energy_threshold,
                        basis_side="right",
                    )
                    a_proj = project_right_null_space(plugin_layer.weight_a, V_cum)
                else:
                    a_proj = plugin_layer.weight_a.clone()

                protected_B_list.append(b_proj)
                protected_A_list.append(a_proj)

                s_p = plugin_layer.scaling * p_weight
                sign = 1.0 if s_p >= 0 else -1.0
                mag = abs(s_p) ** 0.5
                b_blocks.append((sign * mag) * b_proj)
                a_blocks.append(mag * a_proj)

            # Normalise device/dtype before concatenation
            target_device = b_blocks[0].device
            b_blocks = [b.to(device=target_device, dtype=self.torch_dtype) for b in b_blocks]
            a_blocks = [a.to(device=target_device, dtype=self.torch_dtype) for a in a_blocks]

            r_master = b_blocks[0].shape[1]
            total_r = sum(b.shape[1] for b in b_blocks)

            t_r = target_r or self.config.target_r
            if t_r is not None and t_r < total_r:
                if t_r < r_master:
                    raise ValueError(f"target_r ({t_r}) cannot be less than master rank ({r_master})")
                if t_r == r_master or len(b_blocks) == 1:
                    B_final = b_blocks[0]
                    A_final = a_blocks[0]
                    effective_r = r_master
                else:
                    k_plug = t_r - r_master
                    B_plugins = torch.cat(b_blocks[1:], dim=1)
                    A_plugins = torch.cat(a_blocks[1:], dim=0)

                    # Thin QR + small core SVD in O(d * r^2)
                    Q_B, R_B = torch.linalg.qr(B_plugins)
                    Q_A, R_A = torch.linalg.qr(A_plugins.T)

                    M_core = torch.matmul(R_B, R_A.T)
                    U_core, S_core, Vh_core = torch.linalg.svd(M_core, full_matrices=False)

                    k = min(k_plug, len(S_core))
                    sqrt_S = torch.sqrt(torch.clamp(S_core[:k], min=0.0))

                    B_plug_k = torch.matmul(Q_B, U_core[:, :k] * sqrt_S.unsqueeze(0))
                    A_plug_k = torch.matmul(sqrt_S.unsqueeze(1) * Vh_core[:k, :], Q_A.T)

                    B_final = torch.cat([b_blocks[0], B_plug_k], dim=1)
                    A_final = torch.cat([a_blocks[0], A_plug_k], dim=0)
                    effective_r = r_master + k
            else:
                B_final = torch.cat(b_blocks, dim=1)
                A_final = torch.cat(a_blocks, dim=0)
                effective_r = total_r

            peft_scaling = alpha / effective_r if effective_r > 0 else 1.0
            scale_factor = (1.0 / peft_scaling) ** 0.5
            B_peft = B_final * scale_factor
            A_peft = A_final * scale_factor

            module_ranks[mod_key] = effective_r
            ranks_recorded.append(effective_r)

            if mod_key.startswith("base_model.model."):
                prefix = mod_key
            elif mod_key.startswith("base_model."):
                prefix = f"base_model.model.{mod_key[len('base_model.'):]}"
            else:
                prefix = f"base_model.model.{mod_key}"

            adapter_state_dict[f"{prefix}.lora_A.weight"] = A_peft.cpu()
            adapter_state_dict[f"{prefix}.lora_B.weight"] = B_peft.cpu()

        try:
            from safetensors.torch import save_file
            weights_file = out_path / "adapter_model.safetensors"
            save_file(adapter_state_dict, str(weights_file))
            print(f"✓ Saved PEFT adapter weights (safetensors) to: {weights_file}")
        except ImportError:
            weights_file = out_path / "adapter_model.bin"
            torch.save(adapter_state_dict, str(weights_file))
            print(f"✓ Saved PEFT adapter weights (torch binary) to: {weights_file}")

        final_r = max(ranks_recorded) if ranks_recorded else 16
        base_model_name = master.base_model_name_or_path or "unknown"
        config_dict = {
            "auto_mapping": None,
            "base_model_name_or_path": base_model_name,
            "bias": "none",
            "fan_in_fan_out": master.fan_in_fan_out if master else False,
            "inference_mode": True,
            "init_lora_weights": True,
            "lora_alpha": alpha,
            "lora_dropout": 0.0,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": final_r,
            "target_modules": sorted(list(target_modules_set)),
            "task_type": "CAUSAL_LM",
        }
        if len(set(ranks_recorded)) > 1:
            config_dict["rank_pattern"] = module_ranks
            config_dict["alpha_pattern"] = {}
        else:
            config_dict["rank_pattern"] = {}
            config_dict["alpha_pattern"] = {}

        cfg_file = out_path / "adapter_config.json"
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        print(f"✓ Saved PEFT configuration to: {cfg_file}")
