"""06_create_and_inspect_demo.py: Local CPU LoRA Creation, Merging & Weight Inspection.

This script runs 100% on CPU without needing a GPU or internet access:
1. Creates 2 complete local LoRA adapters (Master Math + Plugin Code) with safetensors.
2. Inspects their original weights.
3. Merges them via Two-Sided Asymmetric NSP into a new standalone PEFT adapter.
4. Inspects and prints the merged weights directly from the generated safetensors file.
"""

import json
import sys
from pathlib import Path
from safetensors.torch import load_file, save_file
import torch

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asym_nsp.merger import AsymmetricNSPMerger, MergeConfig


def create_sample_lora(
    output_dir: Path,
    adapter_name: str,
    base_model: str = "meta-llama/Meta-Llama-3-8B",
    num_layers: int = 4,
    d_model: int = 4096,
    d_mlp: int = 11008,
    r: int = 16,
    lora_alpha: float = 32.0,
    seed: int = 42,
):
    """Creates a synthetic yet structurally realistic PEFT LoRA adapter on disk."""
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {}
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    for layer_idx in range(num_layers):
        # Attention modules (d_model x d_model)
        for mod in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            key = f"base_model.model.model.layers.{layer_idx}.self_attn.{mod}"
            state_dict[f"{key}.lora_A.weight"] = torch.randn(r, d_model) * 0.02
            state_dict[f"{key}.lora_B.weight"] = torch.randn(d_model, r) * 0.02

        # MLP modules
        for mod in ["gate_proj", "up_proj"]:
            key = f"base_model.model.model.layers.{layer_idx}.mlp.{mod}"
            state_dict[f"{key}.lora_A.weight"] = torch.randn(r, d_model) * 0.02
            state_dict[f"{key}.lora_B.weight"] = torch.randn(d_mlp, r) * 0.02

        key = f"base_model.model.model.layers.{layer_idx}.mlp.down_proj"
        state_dict[f"{key}.lora_A.weight"] = torch.randn(r, d_mlp) * 0.02
        state_dict[f"{key}.lora_B.weight"] = torch.randn(d_model, r) * 0.02

    # Save safetensors
    weights_path = output_dir / "adapter_model.safetensors"
    save_file(state_dict, str(weights_path))

    # Save adapter_config.json
    config = {
        "auto_mapping": None,
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": r,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }
    with open(output_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return weights_path, len(state_dict)


def main():
    print("=" * 80)
    print("  100% CPU DEMO: CREATE 2 LORAS -> MERGE VIA ASYMMETRIC NSP -> INSPECT WEIGHTS")
    print("=" * 80)

    base_dir = Path("./models_demo")
    master_dir = base_dir / "master_math_lora"
    plugin_dir = base_dir / "plugin_code_lora"
    merged_dir = Path("./outputs/merged_cpu_adapter")

    # -------------------------------------------------------------
    # ШАГ 1: Создаем 2 локальных адаптера на CPU
    # -------------------------------------------------------------
    print("\n[Шаг 1/4] Создаем 2 локальных LoRA-адаптера...")
    w_m_path, count_m = create_sample_lora(master_dir, "Master_Math", seed=42, r=16)
    print(f"  ✓ Master LoRA (Математика): {master_dir} ({count_m} тензоров весов)")

    w_p_path, count_p = create_sample_lora(plugin_dir, "Plugin_Code", seed=100, r=16)
    print(f"  ✓ Plugin LoRA (Код):        {plugin_dir} ({count_p} тензоров весов)")

    # -------------------------------------------------------------
    # ШАГ 2: Инспектируем веса исходных адаптеров
    # -------------------------------------------------------------
    print("\n[Шаг 2/4] Инспекция весов до слияния:")
    sd_m = load_file(str(w_m_path))
    sample_key = "base_model.model.model.layers.0.self_attn.q_proj"
    
    print(f"  • Master A веса: {sample_key}.lora_A.weight -> Форма: {sd_m[sample_key + '.lora_A.weight'].shape}")
    print(f"  • Master B веса: {sample_key}.lora_B.weight -> Форма: {sd_m[sample_key + '.lora_B.weight'].shape}")
    print(f"    Пример первых 3 значений весов Master B:\n    {sd_m[sample_key + '.lora_B.weight'][:3, 0].numpy()}")

    # -------------------------------------------------------------
    # ШАГ 3: Запускаем мерджинг через Asymmetric NSP
    # -------------------------------------------------------------
    print("\n[Шаг 3/4] Запускаем Asymmetric NSP Merging на CPU...")
    config = MergeConfig(
        master_adapter_path=str(master_dir),
        plugin_adapter_paths=[str(plugin_dir)],
        output_dir=str(merged_dir),
        energy_threshold=0.99,
        export_adapter=True,
        device="cpu",
    )
    merger = AsymmetricNSPMerger(config)
    merger.export_as_peft_adapter(output_dir=merged_dir)

    # -------------------------------------------------------------
    # ШАГ 4: Показываем и инспектируем веса смердженного адаптера
    # -------------------------------------------------------------
    print("\n[Шаг 4/4] ИНСПЕКЦИЯ СМЕРДЖЕННЫХ ВЕСОВ В ФАЙЛЕ:")
    merged_weights_file = merged_dir / "adapter_model.safetensors"
    merged_sd = load_file(str(merged_weights_file))

    print(f"\nФайл со смердженными весами: {merged_weights_file}")
    print(f"Всего слоев со смердженными весами: {len(merged_sd)}\n")
    print(f"{'Название слоя весов':<70} | {'Размерность':<20} | {'Тип'}")
    print("-" * 105)

    for i, (name, tensor) in enumerate(merged_sd.items()):
        if i < 10:  # покажем первые 10 тензоров
            print(f"{name:<70} | {str(list(tensor.shape)):<20} | {tensor.dtype}")

    if len(merged_sd) > 10:
        print(f"... и еще {len(merged_sd) - 10} слоев весов.")

    print("\n" + "=" * 80)
    print("✓ ГОТОВО! Смердженный LoRA-адаптер полностью сохранен на диске.")
    print(f"Папка с адаптером: {merged_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
