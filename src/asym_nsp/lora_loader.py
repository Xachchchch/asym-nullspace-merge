"""Loader and parser for HuggingFace PEFT / safetensors LoRA adapters."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch


@dataclass
class LoRALayerWeights:
    """Holds A and B weight matrices and scaling configuration for a single module."""
    module_key: str
    weight_a: torch.Tensor  # [r, d_in]
    weight_b: torch.Tensor  # [d_out, r]
    r: int
    lora_alpha: float = 16.0

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.r if self.r > 0 else 1.0

    def compute_delta_w(self) -> torch.Tensor:
        """Computes the full rank delta weight: scaling * (B @ A)."""
        return self.scaling * torch.matmul(self.weight_b, self.weight_a)


@dataclass
class LoRAAdapter:
    """Represents a complete loaded LoRA adapter with all module layers and metadata."""
    name: str
    config: dict
    layers: Dict[str, LoRALayerWeights] = field(default_factory=dict)
    base_model_name_or_path: Optional[str] = None
    default_r: int = 16
    default_alpha: float = 32.0

    def get_layer(self, module_key: str) -> Optional[LoRALayerWeights]:
        return self.layers.get(module_key)

    def module_keys(self) -> List[str]:
        return sorted(list(self.layers.keys()))


def _normalize_module_key(key: str) -> str:
    """Strips lora_A/lora_B suffixes and prefix wrappers to obtain canonical module name."""
    clean = key
    if clean.startswith("base_model.model."):
        clean = clean[len("base_model.model."):]
    elif clean.startswith("base_model."):
        clean = clean[len("base_model."):]

    # Remove trailing parameter designations
    for suffix in [
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_A.default.weight",
        ".lora_B.default.weight",
        ".weight",
    ]:
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)]
            break

    return clean


def load_state_dict(path_or_dir: Union[str, Path]) -> dict:
    """Loads state dict from safetensors or PyTorch binary file."""
    path = Path(path_or_dir)
    if path.is_dir():
        safetensor_file = path / "adapter_model.safetensors"
        bin_file = path / "adapter_model.bin"
        if safetensor_file.exists():
            path = safetensor_file
        elif bin_file.exists():
            path = bin_file
        else:
            raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin found in {path}")

    if str(path).endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            return load_file(str(path))
        except ImportError:
            raise ImportError("safetensors is required. Install with `pip install safetensors`.")
    else:
        return torch.load(str(path), map_location="cpu")


def load_lora_adapter(
    adapter_path_or_id: Union[str, Path],
    name: Optional[str] = None,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> LoRAAdapter:
    """Loads a LoRA adapter from local directory or HuggingFace hub.

    Args:
        adapter_path_or_id: Local folder path or HuggingFace repo ID
        name: Optional custom identifier for the adapter
        device: Torch device to load tensors onto ('cpu', 'cuda', etc.)
        dtype: Optional cast dtype (torch.float32, torch.bfloat16, etc.)

    Returns:
        LoRAAdapter instance containing parsed layer weights.
    """
    path = Path(adapter_path_or_id)
    adapter_name = name or (path.name if path.exists() else str(adapter_path_or_id).replace("/", "_"))

    config = {}
    config_path = path / "adapter_config.json" if path.is_dir() else None
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    elif not path.exists():
        # Try fetching from huggingface_hub if available
        try:
            from huggingface_hub import snapshot_download
            download_dir = snapshot_download(repo_id=str(adapter_path_or_id))
            path = Path(download_dir)
            cfg_file = path / "adapter_config.json"
            if cfg_file.exists():
                with open(cfg_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
        except Exception as e:
            # Fallback if offline or synthetic
            pass

    state_dict = load_state_dict(path)

    default_r = config.get("r", 16)
    default_alpha = float(config.get("lora_alpha", 32.0))
    base_model = config.get("base_model_name_or_path")

    # Group tensors into A and B pairs per module
    tensors_a: Dict[str, torch.Tensor] = {}
    tensors_b: Dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        tensor = v.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)

        if "lora_A" in k:
            mod_key = _normalize_module_key(k)
            tensors_a[mod_key] = tensor
        elif "lora_B" in k:
            mod_key = _normalize_module_key(k)
            tensors_b[mod_key] = tensor

    layers: Dict[str, LoRALayerWeights] = {}
    common_modules = set(tensors_a.keys()).intersection(set(tensors_b.keys()))

    for mod in common_modules:
        w_a = tensors_a[mod]
        w_b = tensors_b[mod]
        r = w_a.shape[0]
        layers[mod] = LoRALayerWeights(
            module_key=mod,
            weight_a=w_a,
            weight_b=w_b,
            r=r,
            lora_alpha=default_alpha,
        )

    return LoRAAdapter(
        name=adapter_name,
        config=config,
        layers=layers,
        base_model_name_or_path=base_model,
        default_r=default_r,
        default_alpha=default_alpha,
    )
