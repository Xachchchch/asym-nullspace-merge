"""08_run_real_baselines.py: Real-world Baseline Merging (TIES-Merging & DARE) for Qwen2.5-0.5B.

This script implements real parameter merging of two community LoRA adapters:
  1. Math/Reasoning Adapter: xue10/qwen2.5-0.5B-MathInstruct-lora
  2. Multilingual Classifier Adapter: 1shoomun/Multilingual-classifier-qwen-2.5-0.5b-lora
onto the shared base model:
  Base Model: Qwen/Qwen2.5-0.5B

Algorithms Implemented:
  1. TIES-Merging (Yadav et al., NeurIPS 2023):
     - Step 1 (Trim): Prune bottom 80% parameters by magnitude (keep top 20%).
     - Step 2 (Elect Sign): Majority sign voting across trimmed task vectors.
     - Step 3 (Disjoint Merge): Keep only parameters matching the majority sign
       and compute their average across surviving models.
     - Add merged delta weights onto base model and export full HuggingFace model.

  2. DARE (Yu et al., ICML 2024):
     - Step 1 (Drop): Bernoulli dropout with p=0.5 (randomly zero 50% of delta weights).
     - Step 2 (Rescale): Scale surviving parameters by 1 / (1 - p) = 2.0.
     - Step 3 (Merge): Task-arithmetic linear combination (average) of rescaled deltas.
     - Add merged delta weights onto base model and export full HuggingFace model.

Outputs:
  - ./outputs/qwen_ties_merged (Full HuggingFace model + tokenizer)
  - ./outputs/qwen_dare_merged (Full HuggingFace model + tokenizer)
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

# Ensure workspace 'src' is importable
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from asym_nsp.lora_loader import LoRAAdapter, load_lora_adapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("real_baselines")


# ============================================================================
# 1. TIES-Merging Core Functions (Trimming, Sign Election, Disjoint Merge)
# ============================================================================

def trim_by_magnitude(tensor: torch.Tensor, prune_ratio: float = 0.8) -> torch.Tensor:
    """Trims bottom `prune_ratio` of parameters by absolute magnitude.

    Keeps top (1 - prune_ratio) largest parameters; sets all others to 0.

    Args:
        tensor: Task vector delta tensor (any shape).
        prune_ratio: Fraction of elements to drop (e.g. 0.8 = drop bottom 80%).

    Returns:
        Trimmed tensor with identical shape and dtype.
    """
    if prune_ratio <= 0.0:
        return tensor.clone()
    if prune_ratio >= 1.0:
        return torch.zeros_like(tensor)

    abs_tensor = tensor.abs()
    flat_abs = abs_tensor.reshape(-1)
    numel = flat_abs.numel()

    # Fast O(N) selection using kthvalue (10x faster than quantile on CPU)
    k = max(1, min(numel, int(round(prune_ratio * numel))))
    threshold = torch.kthvalue(flat_abs, k).values
    mask = abs_tensor >= threshold

    return tensor * mask


def ties_merge_deltas(
    deltas: List[torch.Tensor],
    prune_ratio: float = 0.8,
    weights: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Applies TIES-Merging across multiple task vector deltas.

    Steps:
      1. Trim: Zero out bottom `prune_ratio` by magnitude for each delta.
      2. Elect Sign: Compute direction of majority mass: sign(sum_i w_i * trimmed_i).
      3. Disjoint Merge: Filter out conflicting signs, then average surviving entries.

    Args:
        deltas: List of delta tensors of the same shape.
        prune_ratio: Fraction of weights to prune per task (default: 0.8).
        weights: Optional task weights (default: equal weighting).

    Returns:
        (merged_delta, stats_dict)
    """
    n_tasks = len(deltas)
    if weights is None:
        weights = [1.0 / n_tasks] * n_tasks

    # Step 1: Trim (Keep top 20% by magnitude)
    trimmed_deltas = [trim_by_magnitude(d, prune_ratio=prune_ratio) for d in deltas]

    # Step 2: Elect Sign
    # Direction of total mass across trimmed task vectors
    mass_sum = torch.zeros_like(deltas[0], dtype=torch.float32)
    for t_delta, w in zip(trimmed_deltas, weights):
        mass_sum += w * t_delta.float()
    majority_sign = torch.sign(mass_sum)

    # Step 3: Disjoint Merge
    disjoint_sum = torch.zeros_like(deltas[0], dtype=torch.float32)
    survivors_count = torch.zeros_like(deltas[0], dtype=torch.float32)

    for t_delta in trimmed_deltas:
        # Agreement condition: sign matches majority sign, majority sign is non-zero, and delta is non-zero
        agree_mask = (torch.sign(t_delta) == majority_sign) & (majority_sign != 0) & (t_delta != 0)
        disjoint_sum += (t_delta * agree_mask).float()
        survivors_count += agree_mask.float()

    # Average surviving disjoint parameters (0 if no adapter survived/agreed)
    merged = torch.where(
        survivors_count > 0,
        disjoint_sum / survivors_count.clamp(min=1.0),
        torch.zeros_like(disjoint_sum),
    ).to(dtype=deltas[0].dtype)

    # Compute summary diagnostics
    numel = deltas[0].numel()
    active_count = (merged != 0).sum().item()
    stats = {
        "active_params_ratio": active_count / max(1, numel),
        "total_params": numel,
    }

    return merged, stats


# ============================================================================
# 2. DARE Core Functions (Bernoulli Dropout + Rescaling + Linear Combination)
# ============================================================================

def dare_merge_deltas(
    deltas: List[torch.Tensor],
    drop_prob: float = 0.5,
    weights: Optional[List[float]] = None,
    seed: Optional[int] = 42,
) -> Tuple[torch.Tensor, dict]:
    """Applies DARE (Drop And REscale) merging across multiple task vectors.

    Steps:
      1. Drop: Sample Bernoulli mask with retention probability (1 - p).
      2. Rescale: Multiply surviving elements by 1 / (1 - p) to preserve expectation.
      3. Linear Merge: Compute weighted linear combination of rescaled task vectors.

    Args:
        deltas: List of delta tensors of identical shape.
        drop_prob: Dropout probability p in Bernoulli sampling (default: 0.5).
        weights: Optional task weights (default: equal weighting).
        seed: Optional RNG seed for reproducible mask generation.

    Returns:
        (merged_delta, stats_dict)
    """
    n_tasks = len(deltas)
    if weights is None:
        weights = [1.0 / n_tasks] * n_tasks

    scale = 1.0 / (1.0 - drop_prob) if drop_prob < 1.0 else 0.0

    rescaled_deltas = []
    rng_generator = torch.Generator(device=deltas[0].device)
    if seed is not None:
        rng_generator.manual_seed(seed)

    for delta in deltas:
        if drop_prob <= 0.0:
            rescaled_deltas.append(delta.clone())
            continue
        if drop_prob >= 1.0:
            rescaled_deltas.append(torch.zeros_like(delta))
            continue

        # Bernoulli mask: 1 with prob (1 - p), 0 with prob p
        mask = (
            torch.rand(delta.shape, generator=rng_generator, device=delta.device) > drop_prob
        ).to(dtype=delta.dtype)

        rescaled = delta * mask * scale
        rescaled_deltas.append(rescaled)

    # Weighted linear combination / task arithmetic
    merged = torch.zeros_like(deltas[0], dtype=torch.float32)
    for rescaled, w in zip(rescaled_deltas, weights):
        merged += w * rescaled.float()

    merged = merged.to(dtype=deltas[0].dtype)

    numel = deltas[0].numel()
    active_count = (merged != 0).sum().item()
    stats = {
        "active_params_ratio": active_count / max(1, numel),
        "total_params": numel,
    }

    return merged, stats


# ============================================================================
# 3. Merging Orchestrator for LoRA Adapters
# ============================================================================

def compute_merged_adapter_deltas(
    adapters: List[LoRAAdapter],
    method: str = "ties",
    prune_ratio: float = 0.8,
    drop_prob: float = 0.5,
    weights: Optional[List[float]] = None,
    seed: Optional[int] = 42,
) -> Dict[str, torch.Tensor]:
    """Computes merged delta weights for all common modules across adapters.

    Args:
        adapters: List of loaded LoRAAdapter instances.
        method: "ties" or "dare".
        prune_ratio: Prune ratio for TIES (default: 0.8).
        drop_prob: Dropout probability for DARE (default: 0.5).
        weights: Weights for adapters.
        seed: Random seed for DARE.

    Returns:
        Dictionary mapping canonical module name to merged delta_W tensor.
    """
    assert len(adapters) >= 2, "At least 2 adapters are required for merging."

    # Identify common module keys across all adapters
    common_keys = set(adapters[0].module_keys())
    for ad in adapters[1:]:
        common_keys = common_keys.intersection(set(ad.module_keys()))
    sorted_keys = sorted(list(common_keys))

    logger.info(
        f"Found {len(sorted_keys)} intersecting module layers across {len(adapters)} adapters."
    )

    merged_deltas: Dict[str, torch.Tensor] = {}

    for mod_key in tqdm(sorted_keys, desc=f"Computing {method.upper()} merged deltas"):
        # Extract full-rank delta weights: Delta_W = scaling * (B @ A)
        task_deltas = []
        for ad in adapters:
            layer = ad.get_layer(mod_key)
            if layer is None:
                continue
            delta_w = layer.compute_delta_w()
            task_deltas.append(delta_w)

        if len(task_deltas) != len(adapters):
            continue

        if method.lower() == "ties":
            merged, _ = ties_merge_deltas(
                task_deltas,
                prune_ratio=prune_ratio,
                weights=weights,
            )
        elif method.lower() == "dare":
            merged, _ = dare_merge_deltas(
                task_deltas,
                drop_prob=drop_prob,
                weights=weights,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown merging method: {method}. Choose 'ties' or 'dare'.")

        merged_deltas[mod_key] = merged

    return merged_deltas


# ============================================================================
# 4. Applying Merged Deltas to Base Model & Saving as Full HF Model
# ============================================================================

def apply_deltas_and_save(
    base_model_id_or_path: str,
    merged_deltas: Dict[str, torch.Tensor],
    output_dir: Union[str, Path],
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float32,
    save_tokenizer: bool = True,
):
    """Loads base model, applies delta weights, and saves full HF model + tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base model: {base_model_id_or_path} (dtype={torch_dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id_or_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device != "cpu" and torch.cuda.is_available() else None,
    )

    named_params = dict(model.named_parameters())
    applied_count = 0

    for mod_key, delta_w in tqdm(merged_deltas.items(), desc="Applying deltas to base model"):
        # Attempt canonical key match patterns:
        # e.g. "model.layers.0.self_attn.q_proj.weight" or "layers.0.self_attn.q_proj.weight"
        candidate_keys = [
            f"{mod_key}.weight",
            mod_key,
            f"model.{mod_key}.weight",
            f"model.{mod_key}",
        ]

        target_param = None
        for cand in candidate_keys:
            if cand in named_params:
                target_param = named_params[cand]
                break

        # Fallback suffix match if direct candidate wasn't found
        if target_param is None:
            for name, param in named_params.items():
                if name.endswith(f"{mod_key}.weight") or name == mod_key:
                    target_param = param
                    break

        if target_param is not None:
            with torch.no_grad():
                target_param.data.add_(
                    delta_w.to(device=target_param.device, dtype=target_param.dtype)
                )
            applied_count += 1
        else:
            logger.warning(f"No matching base model parameter found for module: {mod_key}")

    logger.info(
        f"Successfully applied delta weights to {applied_count}/{len(merged_deltas)} modules."
    )

    logger.info(f"Saving merged model to {out_path}...")
    model.save_pretrained(out_path)

    if save_tokenizer:
        logger.info(f"Saving tokenizer from {base_model_id_or_path} to {out_path}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id_or_path)
        tokenizer.save_pretrained(out_path)

    logger.info(f"✓ Model successfully saved to {out_path}")

    # Clean up memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# 5. Optional Sanity Generation Check
# ============================================================================

def run_sanity_check(
    model_dir: Union[str, Path],
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float32,
):
    """Runs short inference generations to verify model validity."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Running sanity check for {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        device_map="auto" if device != "cpu" and torch.cuda.is_available() else None,
    )

    test_prompts = [
        "Problem: A car travels 120 miles in 2 hours. What is its speed in mph? Solution:",
        "Classify the sentiment of the following sentence: 'This service is truly exceptional and delightful!' Sentiment:",
    ]

    for p in test_prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print(f"\n[Prompt]: {p}")
        print(f"[Generated]: {generated_text.strip()}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# 6. CLI Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run real-world baseline merging (TIES and DARE) on Qwen2.5-0.5B LoRA adapters."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Base model repository ID or path (default: Qwen/Qwen2.5-0.5B).",
    )
    parser.add_argument(
        "--math_adapter",
        type=str,
        default="xue10/qwen2.5-0.5B-MathInstruct-lora",
        help="First LoRA adapter: MathInstruct (default: xue10/qwen2.5-0.5B-MathInstruct-lora).",
    )
    parser.add_argument(
        "--classifier_adapter",
        type=str,
        default="1shoomun/Multilingual-classifier-qwen-2.5-0.5b-lora",
        help="Second LoRA adapter: Multilingual Classifier (default: 1shoomun/Multilingual-classifier-qwen-2.5-0.5b-lora).",
    )
    parser.add_argument(
        "--ties_output_dir",
        type=str,
        default="./outputs/qwen_ties_merged",
        help="Output directory for TIES-merged full HuggingFace model.",
    )
    parser.add_argument(
        "--dare_output_dir",
        type=str,
        default="./outputs/qwen_dare_merged",
        help="Output directory for DARE-merged full HuggingFace model.",
    )
    parser.add_argument(
        "--ties_prune_ratio",
        type=float,
        default=0.8,
        help="Prune ratio for TIES (default: 0.8 = prune bottom 80%% by magnitude).",
    )
    parser.add_argument(
        "--dare_drop_prob",
        type=float,
        default=0.5,
        help="Dropout probability for DARE (default: 0.5 = 50%% Bernoulli drop).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["both", "ties", "dare"],
        default="both",
        help="Which baseline merge method to execute (default: both).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation and model loading ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype for base model weights (default: float32).",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        default=False,
        help="Compute merged deltas without saving full models to disk.",
    )
    parser.add_argument(
        "--test_inference",
        action="store_true",
        default=False,
        help="Run short sample generation on merged models after saving.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = getattr(torch, args.torch_dtype)

    print("=" * 78)
    print("  REAL-WORLD BASELINE MERGING: TIES-MERGING & DARE FOR QWEN2.5-0.5B")
    print("=" * 78)
    print(f"Base Model:             {args.base_model}")
    print(f"LoRA Adapter 1 (Math):  {args.math_adapter}")
    print(f"LoRA Adapter 2 (Class): {args.classifier_adapter}")
    print(f"Target Method:          {args.method.upper()}")
    print(f"TIES Prune Ratio:       {args.ties_prune_ratio} (Drop 80% lowest magnitude)")
    print(f"DARE Drop Prob:         {args.dare_drop_prob} (50% Bernoulli drop, 2x scale)")
    print(f"TIES Output:            {args.ties_output_dir}")
    print(f"DARE Output:            {args.dare_output_dir}")
    print(f"Device:                 {args.device} (Dtype: {args.torch_dtype})")
    print("=" * 78)

    # 1. Load both real LoRA adapters
    logger.info("[1/3] Loading LoRA adapters...")
    adapter_math = load_lora_adapter(
        args.math_adapter,
        name="math_instruct",
        device="cpu",  # CPU is optimal for low-rank matrix algebra
        dtype=torch.float32,
    )
    logger.info(f"Loaded Math Adapter: {len(adapter_math.layers)} layers.")

    adapter_classifier = load_lora_adapter(
        args.classifier_adapter,
        name="multilingual_classifier",
        device="cpu",
        dtype=torch.float32,
    )
    logger.info(f"Loaded Classifier Adapter: {len(adapter_classifier.layers)} layers.")

    adapters = [adapter_math, adapter_classifier]

    # 2. TIES-Merging
    if args.method in ["both", "ties"]:
        print("\n" + "-" * 78)
        print("  STEP: EXECUTING TIES-MERGING")
        print("-" * 78)
        ties_deltas = compute_merged_adapter_deltas(
            adapters=adapters,
            method="ties",
            prune_ratio=args.ties_prune_ratio,
        )
        logger.info(f"TIES merged deltas computed for {len(ties_deltas)} modules.")

        if not args.skip_save:
            logger.info(f"Applying TIES merged deltas and saving to {args.ties_output_dir}...")
            apply_deltas_and_save(
                base_model_id_or_path=args.base_model,
                merged_deltas=ties_deltas,
                output_dir=args.ties_output_dir,
                device=args.device,
                torch_dtype=dtype,
                save_tokenizer=True,
            )
            if args.test_inference:
                run_sanity_check(args.ties_output_dir, device=args.device, torch_dtype=dtype)
        else:
            logger.info("[!] Skipping disk save as requested (--skip_save).")

    # 3. DARE Merging
    if args.method in ["both", "dare"]:
        print("\n" + "-" * 78)
        print("  STEP: EXECUTING DARE MERGING")
        print("-" * 78)
        dare_deltas = compute_merged_adapter_deltas(
            adapters=adapters,
            method="dare",
            drop_prob=args.dare_drop_prob,
        )
        logger.info(f"DARE merged deltas computed for {len(dare_deltas)} modules.")

        if not args.skip_save:
            logger.info(f"Applying DARE merged deltas and saving to {args.dare_output_dir}...")
            apply_deltas_and_save(
                base_model_id_or_path=args.base_model,
                merged_deltas=dare_deltas,
                output_dir=args.dare_output_dir,
                device=args.device,
                torch_dtype=dtype,
                save_tokenizer=True,
            )
            if args.test_inference:
                run_sanity_check(args.dare_output_dir, device=args.device, torch_dtype=dtype)
        else:
            logger.info("[!] Skipping disk save as requested (--skip_save).")

    print("\n" + "=" * 78)
    print("✓ BASELINE MERGING PIPELINE COMPLETED SUCCESSFULLY!")
    if not args.skip_save:
        if args.method in ["both", "ties"]:
            print(f"  • TIES Merged Model: {args.ties_output_dir}")
        if args.method in ["both", "dare"]:
            print(f"  • DARE Merged Model: {args.dare_output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
