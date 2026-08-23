"""03_eval_benchmark.py: Evaluation script for merged models via lm-evaluation-harness."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate base, individual, and merged models across standard benchmarks."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to merged model checkpoint directory or HuggingFace ID.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,humaneval,mmlu",
        help="Comma-separated benchmark tasks (e.g. 'gsm8k,humaneval,mmlu').",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples for evaluation.",
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
    return parser.parse_args()


def run_evaluation(model_path: str, tasks: list, num_fewshot: int, batch_size: int, output_file: str):
    print("=" * 70)
    print("  EVALUATION BENCHMARK: ASYMMETRIC NSP")
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

    except ImportError:
        print("\n[!] lm-evaluation-harness is not installed.")
        print("To run benchmarks, install the evaluation dependencies:")
        print("    pip install lm-eval datasets accelerate")


if __name__ == "__main__":
    args = parse_args()
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    run_evaluation(
        model_path=args.model_path,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        output_file=args.output_file,
    )
