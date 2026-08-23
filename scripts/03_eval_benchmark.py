"""03_eval_benchmark.py: Evaluation script for merged models.

Supports:
1. Quick text generation & side-by-side prompt testing (Pure PyTorch / Transformers).
2. Perplexity computation on custom evaluation datasets.
3. Formal standard benchmarks via lm-evaluation-harness (GSM8k, HumanEval, MMLU).
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional
import torch


SAMPLE_PROMPTS = {
    "math_reasoning": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's solve step by step:",
    "code_generation": "def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is a prime number, else False.\"\"\"\n",
}


def evaluate_generation(
    model_path: str,
    prompts: dict,
    max_new_tokens: int = 128,
    device: str = "cpu",
):
    """Runs generation test across sample domain prompts."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 70)
    print(f"  RUNNING SAMPLE GENERATION EVALUATION: {model_path}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    results = {}
    for category, prompt in prompts.items():
        print(f"\n--- Category: {category} ---")
        print(f"Prompt:\n{prompt.strip()}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nResponse:\n{generated_text[len(prompt):].strip()}")
        results[category] = {
            "prompt": prompt,
            "response": generated_text[len(prompt):].strip(),
        }

    return results


def run_lm_eval(
    model_path: str,
    tasks: List[str],
    num_fewshot: int,
    batch_size: int,
    output_file: str,
):
    """Runs lm-evaluation-harness benchmarks if installed."""
    print("=" * 70)
    print("  LM-EVALUATION-HARNESS BENCHMARK")
    print("=" * 70)
    print(f"Model:     {model_path}")
    print(f"Tasks:     {tasks}")
    print(f"Few-shot:  {num_fewshot}")

    try:
        import lm_eval
        from lm_eval import evaluator

        print("\nRunning lm-eval benchmarks...")
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path},trust_remote_code=True",
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
        )

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results.get("results", results), f, indent=2)

        print(f"\n✓ Results saved to {out_path}")
        print(json.dumps(results.get("results", {}), indent=2))
        return results

    except ImportError:
        print("\n[!] `lm-eval` is not installed.")
        print("To run formal lm-eval benchmarks, install dependencies:")
        print("    pip install lm-eval datasets accelerate")
        print("\nAlternatively, run quick generation evaluation without lm-eval:")
        print(f"    python scripts/03_eval_benchmark.py --model_path {model_path} --mode generate")
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate base, individual, and merged models across benchmarks."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to merged model directory or HuggingFace repo ID.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "lm_eval"],
        default="generate",
        help="Evaluation mode: 'generate' for sample tests, 'lm_eval' for formal harness benchmarks.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,humaneval,mmlu",
        help="Comma-separated benchmark tasks for lm_eval mode.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples for lm_eval mode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./eval_results.json",
        help="Path to save output JSON metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device ('cpu' or 'cuda').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "generate":
        evaluate_generation(
            model_path=args.model_path,
            prompts=SAMPLE_PROMPTS,
            device=args.device,
        )
    else:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
        run_lm_eval(
            model_path=args.model_path,
            tasks=task_list,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            output_file=args.output_file,
        )


if __name__ == "__main__":
    main()
