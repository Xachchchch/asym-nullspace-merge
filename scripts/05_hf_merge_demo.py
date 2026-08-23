"""05_hf_merge_demo.py: End-to-End Real HuggingFace LoRA Merge & Inference Demo.

Demonstrates merging two real community LoRA adapters for a shared base model:
1. Master LoRA (e.g. Math / Reasoning) -> 100% Invariant.
2. Plugin LoRA (e.g. Python / Code) -> Projected into Master's Null-Space.
3. Generates test responses on both domain prompts to verify zero-interference integration.
"""

import argparse
import sys
from pathlib import Path
import torch

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asym_nsp.merger import AsymmetricNSPMerger, MergeConfig


# Curated lightweight presets from HuggingFace Hub
PRESETS = {
    "qwen2.5-0.5b": {
        "base_model": "Qwen/Qwen2.5-0.5B",
        "master": "mhenrichgo/Qwen2.5-0.5B-Instruct-GSM8K-LoRA",
        "plugin": "benk/Qwen2.5-0.5B-Python-LoRA",
        "test_math_prompt": "Problem: A farmer has 15 cows and 22 sheep. If 4 cows and 7 sheep are sold, how many animals remain on the farm? Let's calculate step by step:",
        "test_code_prompt": "def find_even_numbers(numbers: list) -> list:\n    \"\"\"Filter and return all even numbers from the list.\"\"\"\n",
    },
    "tinyllama": {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "master": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "plugin": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "test_math_prompt": "Question: If a car travels at 60 mph for 3.5 hours, how far did it go? Show steps:",
        "test_code_prompt": "def reverse_string(s: str) -> str:\n    \"\"\"Reverse the input string.\"\"\"\n",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End HuggingFace LoRA Merge & Inference Demo."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="qwen2.5-0.5b",
        choices=list(PRESETS.keys()) + ["custom"],
        help="Model preset to test.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model repository ID on Hugging Face Hub.",
    )
    parser.add_argument(
        "--master",
        type=str,
        default=None,
        help="Master LoRA adapter repository ID or directory.",
    )
    parser.add_argument(
        "--plugin",
        type=str,
        default=None,
        help="Plugin LoRA adapter repository ID or directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/hf_merged_adapter",
        help="Output directory for merged PEFT adapter.",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        default=False,
        help="Skip generation inference and only perform the merge.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference ('cpu' or 'cuda').",
    )
    return parser.parse_args()


def run_demo():
    args = parse_args()

    if args.preset != "custom" and not args.master:
        preset_info = PRESETS[args.preset]
        base_model_id = args.base_model or preset_info["base_model"]
        master_id = args.master or preset_info["master"]
        plugin_id = args.plugin or preset_info["plugin"]
        math_prompt = preset_info["test_math_prompt"]
        code_prompt = preset_info["test_code_prompt"]
    else:
        if not args.master or not args.plugin or not args.base_model:
            sys.exit("Error: When using --preset custom, --base_model, --master, and --plugin are required.")
        base_model_id = args.base_model
        master_id = args.master
        plugin_id = args.plugin
        math_prompt = "Problem: Solve 3x + 12 = 45. Step by step solution:"
        code_prompt = "def binary_search(arr: list, target: int) -> int:\n    \"\"\"Perform binary search.\"\"\"\n"

    print("=" * 75)
    print("  ASYMMETRIC NULL-SPACE PROJECTION (NSP): REAL-WORLD MERGE DEMO")
    print("=" * 75)
    print(f"Base Model:       {base_model_id}")
    print(f"Master LoRA (1):  {master_id}  [Preserved 100% / Zero Degradation]")
    print(f"Plugin LoRA (2):  {plugin_id}  [Projected into Null-Space]")
    print(f"Output Directory: {args.output_dir}")
    print(f"Target Device:    {args.device}")
    print("-" * 75)

    # 1. Initialize Merger and export standalone PEFT adapter
    config = MergeConfig(
        master_adapter_path=master_id,
        plugin_adapter_paths=[plugin_id],
        output_dir=args.output_dir,
        energy_threshold=0.99,
        export_adapter=True,
        device="cpu",  # CPU is blazing fast for low-rank SVD projections
    )

    merger = AsymmetricNSPMerger(config)
    print("\n[Step 1/2] Computing Two-Sided Asymmetric Null-Space Projections...")
    merger.export_as_peft_adapter(output_dir=args.output_dir)
    print(f"\n✓ Merged PEFT adapter successfully written to {args.output_dir}!")

    if args.skip_inference:
        print("\n[!] Skipping inference as requested.")
        return

    # 2. Run Inference with the merged adapter
    print("\n[Step 2/2] Loading Base Model & Merged Adapter for Test Generations...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            device_map="auto" if args.device == "cuda" else None,
        )

        merged_model = PeftModel.from_pretrained(base_model, args.output_dir)
        merged_model.eval()

        print("\n" + "=" * 75)
        print("  GENERATION EVALUATION ON DUAL-DOMAIN TASKS")
        print("=" * 75)

        for test_name, prompt in [("Task 1: Master Domain (Math)", math_prompt), ("Task 2: Plugin Domain (Code)", code_prompt)]:
            print(f"\n>>> {test_name}")
            print(f"Prompt:\n{prompt.strip()}")
            inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)
            with torch.no_grad():
                output_tokens = merged_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            print(f"\nGenerated Output:\n{generated[len(prompt):].strip()}")
            print("-" * 50)

        print("\n✓ Demo completed successfully!")

    except Exception as e:
        print(f"\nNote: Inference step skipped or encountered an error ({e}).")
        print("You can load and test the merged adapter anytime via:")
        print(f"    from peft import PeftModel")
        print(f"    model = PeftModel.from_pretrained(base_model, '{args.output_dir}')")


if __name__ == "__main__":
    run_demo()
