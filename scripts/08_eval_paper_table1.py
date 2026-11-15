"""08_eval_paper_table1.py
===========================================================
Reproducible Cloud GPU Evaluation of Paper Table 1 via lm-evaluation-harness.

Evaluates and compares:
  1. isolated_master       – Base model + Master LoRA
  2. isolated_plugin       – Base model + Plugin LoRA
  3. naive                 – Elementwise addition of delta weights
  4. task_arithmetic       – Scaled sum (Ilharco et al., 2023)
  5. ties                  – TIES-Merging (Yadav et al., 2023)
  6. dare                  – DARE with Task Arithmetic (Yu et al., 2024)
  7. asym_nsp_output_only  – Asymmetric NSP Output-Only (project_left=True)
  8. asym_nsp_two_sided    – Asymmetric NSP Two-Sided (project_left=True, project_right=True)

Usage:
  # Quick dry-run (prints plan without loading weights or requiring lm_eval):
  python scripts/08_eval_paper_table1.py \
      --base_model Qwen/Qwen2.5-0.5B \
      --master xue10/Qwen2.5-0.5B-Math-LoRA \
      --plugins 1shoomun/Qwen2.5-0.5B-LoRA \
      --dry_run

  # Full evaluation on Cloud GPU:
  python scripts/08_eval_paper_table1.py \
      --base_model Qwen/Qwen2.5-0.5B \
      --master xue10/Qwen2.5-0.5B-Math-LoRA \
      --plugins 1shoomun/Qwen2.5-0.5B-LoRA \
      --tasks gsm8k,arc_easy \
      --device cuda
===========================================================
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asym_nsp.baselines import merge_lora_adapters_baseline
from asym_nsp.lora_loader import load_lora_adapter
from asym_nsp.merger import AsymmetricNSPMerger, MergeConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproducible Cloud GPU Evaluation of Paper Table 1 via lm-evaluation-harness."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model path or HuggingFace ID (e.g. 'Qwen/Qwen2.5-0.5B').",
    )
    parser.add_argument(
        "--master",
        type=str,
        required=True,
        help="Master LoRA adapter path or HuggingFace Hub ID.",
    )
    parser.add_argument(
        "--plugins",
        nargs="+",
        type=str,
        required=True,
        help="One or more Plugin LoRA adapter paths or HuggingFace Hub IDs.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,arc_easy",
        help="Comma-separated list of lm-eval tasks (default: 'gsm8k,arc_easy').",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="isolated_master,isolated_plugin,naive,task_arithmetic,ties,dare,asym_nsp_output_only,asym_nsp_two_sided",
        help="Comma-separated list of methods to evaluate.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples for lm-eval (default: 0).",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="auto",
        help="Batch size for lm-eval (default: 'auto').",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional limit on number of examples per task (for quick sanity checks; None runs full split).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype passed to lm_eval model_args (default: 'auto').",
    )
    parser.add_argument(
        "--energy_threshold",
        type=float,
        default=1.0,
        help="Energy threshold for Asym-NSP (default: 1.0 for 100%% preservation).",
    )
    parser.add_argument(
        "--ta_lam",
        type=float,
        default=0.5,
        help="Scaling lambda for Task Arithmetic (default: 0.5).",
    )
    parser.add_argument(
        "--ties_top_k",
        type=float,
        default=20.0,
        help="Top-k percentage for TIES trimming (default: 20.0).",
    )
    parser.add_argument(
        "--ties_lam",
        type=float,
        default=1.0,
        help="Scaling lambda for TIES disjoint merge (default: 1.0).",
    )
    parser.add_argument(
        "--dare_drop_rate",
        type=float,
        default=0.5,
        help="Drop rate p for DARE (default: 0.5).",
    )
    parser.add_argument(
        "--dare_lam",
        type=float,
        default=0.5,
        help="Scaling lambda for DARE base method (default: 0.5).",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="./outputs/table1_eval",
        help="Directory to store intermediate merged weights and standalone adapters.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="./outputs/table1_eval/results.json",
        help="Path to save incremental JSON evaluation results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip evaluating methods already present in --output_json.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print full evaluation plan and exit cleanly without downloading weights or requiring lm_eval.",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Metric Formatting & Markdown Table
# ══════════════════════════════════════════════════════════════════════════════

def extract_task_metric(task_res: dict) -> Tuple[float, Optional[float], str]:
    """Finds the primary score and stderr from an lm-eval task result dict."""
    candidate_keys = [
        "exact_match,none",
        "acc_norm,none",
        "acc,none",
        "exact_match",
        "acc_norm",
        "acc",
        "pass@1",
        "f1",
    ]
    for key in candidate_keys:
        if key in task_res:
            val = float(task_res[key])
            stderr_key = key.replace(",none", "_stderr,none") if ",none" in key else f"{key}_stderr"
            stderr = float(task_res[stderr_key]) if stderr_key in task_res and task_res[stderr_key] is not None else None
            metric_name = key.split(",")[0]
            return val, stderr, metric_name

    for k, v in task_res.items():
        if isinstance(v, (int, float)) and "stderr" not in k:
            stderr_k = f"{k}_stderr"
            stderr = float(task_res[stderr_k]) if stderr_k in task_res and task_res[stderr_k] is not None else None
            return float(v), stderr, k

    return 0.0, None, "score"


def format_score(score: float, stderr: Optional[float]) -> str:
    """Formats score as percentage with optional stderr."""
    pct = score * 100.0 if score <= 1.0 else score
    if stderr is not None:
        err_pct = stderr * 100.0 if score <= 1.0 else stderr
        return f"{pct:.2f} ± {err_pct:.2f}%"
    return f"{pct:.2f}%"


def format_markdown_table(results_dict: dict, tasks: List[str], master_task: str) -> str:
    """Renders formatted Markdown summary table."""
    header = ["| Method |"] + [f" {t} |" for t in tasks] + [" Master Retention (%) |", " Master Δ |"]
    sep = ["|:--- |"] + [":---: |" for _ in tasks] + [":---: |", ":---: |"]
    lines = ["".join(header), "".join(sep)]

    isolated_master_score = None
    if "isolated_master" in results_dict:
        m_task_data = results_dict["isolated_master"].get(master_task, {})
        isolated_master_score = m_task_data.get("score")

    for method, m_data in results_dict.items():
        row = [f"| **{method}** |"]
        for t in tasks:
            t_data = m_data.get(t, {})
            score = t_data.get("score")
            stderr = t_data.get("stderr")
            if score is not None:
                row.append(f" {format_score(score, stderr)} |")
            else:
                row.append(" N/A |")

        m_score = m_data.get(master_task, {}).get("score")
        if isolated_master_score is not None and m_score is not None:
            retention = (m_score / (isolated_master_score + 1e-12)) * 100.0
            diff = (m_score - isolated_master_score) * 100.0 if isolated_master_score <= 1.0 else (m_score - isolated_master_score)
            row.append(f" {retention:.2f}% |")
            row.append(f" {diff:+.2f}% |")
        else:
            row.append(" N/A |")
            row.append(" N/A |")

        lines.append("".join(row))

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Dry Run Execution Plan
# ══════════════════════════════════════════════════════════════════════════════

def print_dry_run_plan(args, methods: List[str], tasks: List[str]):
    """Prints dry-run evaluation plan and exits cleanly."""
    print("=" * 80)
    print("  PAPER TABLE 1 EVALUATION PLAN (DRY RUN)")
    print("=" * 80)
    print(f"Base Model:        {args.base_model}")
    print(f"Master LoRA:       {args.master}")
    print(f"Plugin LoRA(s):    {args.plugins}")
    print(f"Evaluation Tasks:  {tasks}")
    print(f"Methods to Eval:   {methods}")
    print(f"Few-shot:          {args.num_fewshot}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Limit:             {args.limit if args.limit is not None else 'Full split (None)'}")
    print(f"Device:            {args.device}")
    print(f"Dtype:             {args.dtype}")
    print(f"Energy Threshold:  {args.energy_threshold}")
    print(f"Hyperparameters:   TA λ={args.ta_lam}, TIES top-k={args.ties_top_k}%, λ={args.ties_lam}, DARE drop={args.dare_drop_rate}, λ={args.dare_lam}")
    print(f"Work Directory:    {args.work_dir}")
    print(f"Output JSON:       {args.output_json}")
    print(f"Skip Existing:     {args.skip_existing}")
    print("-" * 80)
    print("METHOD-BY-METHOD EXECUTION PLAN:")

    for idx, m in enumerate(methods, start=1):
        print(f"\n[{idx}/{len(methods)}] Method: {m}")
        if m == "isolated_master":
            print(f"  • Strategy:   Evaluate base model with unmerged Master LoRA adapter")
            print(f"  • Model Arg:  pretrained={args.base_model},peft={args.master}")
        elif m == "isolated_plugin":
            print(f"  • Strategy:   Evaluate base model with unmerged Plugin LoRA adapter")
            print(f"  • Model Arg:  pretrained={args.base_model},peft={args.plugins[0]}")
        elif m in ("asym_nsp_output_only", "asym_nsp_two_sided"):
            proj_left = True
            proj_right = (m == "asym_nsp_two_sided")
            peft_out = Path(args.work_dir) / m
            print(f"  • Strategy:   Asymmetric NSP Factor-Space Merge (project_left={proj_left}, project_right={proj_right}, energy_threshold={args.energy_threshold})")
            print(f"  • Action:     Export standalone PEFT adapter to {peft_out}")
            print(f"  • Model Arg:  pretrained={args.base_model},peft={peft_out}")
        elif m in ("naive", "task_arithmetic", "ties", "dare"):
            merged_out = Path(args.work_dir) / m
            hparam_str = ""
            if m == "task_arithmetic":
                hparam_str = f" (λ={args.ta_lam})"
            elif m == "ties":
                hparam_str = f" (top_k={args.ties_top_k}%, λ={args.ties_lam})"
            elif m == "dare":
                hparam_str = f" (drop_rate={args.dare_drop_rate}, λ={args.dare_lam})"
            print(f"  • Strategy:   Baseline Delta Merge: {m}{hparam_str}")
            print(f"  • Action:     Merge deltas via merge_lora_adapters_baseline and apply_and_save to {merged_out}")
            print(f"  • Model Arg:  pretrained={merged_out}")
        else:
            print(f"  • Strategy:   Custom evaluation mode: {m}")

    print("\n" + "=" * 80)
    print("✓ Dry run plan generated successfully. Exiting cleanly (code 0).")
    print("=" * 80)
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
# Main Evaluation Loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    master_task = tasks[0] if tasks else "gsm8k"

    if args.dry_run:
        print_dry_run_plan(args, methods, tasks)

    # Verify lm_eval is available before proceeding
    try:
        import lm_eval
        from lm_eval import evaluator
    except ImportError:
        print("\n[!] `lm-eval` is not installed.")
        print("To run formal Paper Table 1 benchmarks, please install dependencies:")
        print("    pip install lm-eval datasets accelerate\n")
        sys.exit(0)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if resuming
    results_dict: Dict[str, dict] = {}
    if args.skip_existing and out_json.exists():
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                saved = json.load(f)
                results_dict = saved.get("methods", {})
                print(f"Loaded existing results for {len(results_dict)} methods from {out_json}")
        except Exception as e:
            print(f"Warning: could not read existing JSON ({e}), starting fresh.")

    def run_eval(model_path: str, peft_adapter: Optional[str] = None):
        """Runs simple_evaluate on specified model and optional peft adapter."""
        model_args = f"pretrained={model_path},trust_remote_code=True"
        if peft_adapter:
            model_args += f",peft={peft_adapter}"
        if args.dtype != "auto":
            model_args += f",dtype={args.dtype}"
        if args.device != "cuda":
            model_args += f",device={args.device}"

        eval_kwargs = {
            "model": "hf",
            "model_args": model_args,
            "tasks": tasks,
            "num_fewshot": args.num_fewshot,
            "batch_size": int(args.batch_size) if args.batch_size.isdigit() else args.batch_size,
        }
        if args.limit is not None:
            eval_kwargs["limit"] = int(args.limit) if args.limit.is_integer() else args.limit

        return evaluator.simple_evaluate(**eval_kwargs)

    # Iterate over methods
    for idx, method in enumerate(methods, start=1):
        print("\n" + "=" * 80)
        print(f"  [{idx}/{len(methods)}] EVALUATING METHOD: {method}")
        print("=" * 80)

        if args.skip_existing and method in results_dict:
            print(f"Skipping {method} (already present in {out_json})")
            continue

        raw_eval_output = None

        if method == "isolated_master":
            raw_eval_output = run_eval(args.base_model, peft_adapter=args.master)

        elif method == "isolated_plugin":
            raw_eval_output = run_eval(args.base_model, peft_adapter=args.plugins[0])

        elif method in ("asym_nsp_output_only", "asym_nsp_two_sided"):
            proj_left = True
            proj_right = (method == "asym_nsp_two_sided")
            peft_dir = work_dir / method
            peft_dir.mkdir(parents=True, exist_ok=True)

            print(f"Merging and exporting standalone PEFT adapter to {peft_dir}...")
            cfg = MergeConfig(
                master_adapter_path=args.master,
                plugin_adapter_paths=args.plugins,
                output_dir=str(peft_dir),
                energy_threshold=args.energy_threshold,
                project_left=proj_left,
                project_right=proj_right,
                device=args.device,
                dtype="float32",
            )
            merger = AsymmetricNSPMerger(cfg)
            merger.export_as_peft_adapter(peft_dir)
            raw_eval_output = run_eval(args.base_model, peft_adapter=str(peft_dir))

        elif method in ("naive", "task_arithmetic", "ties", "dare"):
            merged_dir = work_dir / method
            merged_dir.mkdir(parents=True, exist_ok=True)

            print(f"Loading LoRA adapters for baseline merge: {method}...")
            master_adapter = load_lora_adapter(args.master, device="cpu", dtype=torch.float32)
            plugin_adapters = [
                load_lora_adapter(p, device="cpu", dtype=torch.float32)
                for p in args.plugins
            ]

            kwargs = {}
            if method == "task_arithmetic":
                kwargs["scaling_lam"] = args.ta_lam
            elif method == "ties":
                kwargs["top_k_pct"] = args.ties_top_k
                kwargs["scaling_lam"] = args.ties_lam
            elif method == "dare":
                kwargs["drop_rate"] = args.dare_drop_rate
                kwargs["scaling_lam"] = args.dare_lam
                kwargs["base_method"] = "task_arithmetic"

            merged_deltas = merge_lora_adapters_baseline(
                master=master_adapter,
                plugins=plugin_adapters,
                method=method,
                **kwargs,
            )

            cfg = MergeConfig(
                master_adapter_path=args.master,
                plugin_adapter_paths=args.plugins,
                device=args.device,
            )
            merger = AsymmetricNSPMerger(cfg)
            merger.apply_and_save(
                base_model_path=args.base_model,
                output_dir=merged_dir,
                merged_deltas=merged_deltas,
                master=master_adapter,
            )
            raw_eval_output = run_eval(str(merged_dir))

        else:
            print(f"Warning: Unknown method '{method}'. Skipping.")
            continue

        # Parse task results
        method_results = {}
        if raw_eval_output and "results" in raw_eval_output:
            for t in tasks:
                t_raw = raw_eval_output["results"].get(t, {})
                score, stderr, mname = extract_task_metric(t_raw)
                method_results[t] = {
                    "score": score,
                    "stderr": stderr,
                    "metric_name": mname,
                }
        results_dict[method] = method_results

        # Save incremental results to JSON
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "base_model": args.base_model,
                "master": args.master,
                "plugins": args.plugins,
                "tasks": tasks,
                "methods": results_dict,
            }, f, indent=2)

        # Print current summary table
        print("\n" + "-" * 70)
        print("  INCREMENTAL RESULTS SUMMARY (Markdown)")
        print("-" * 70)
        print(format_markdown_table(results_dict, tasks, master_task))
        print("-" * 70)

    print("\n" + "=" * 80)
    print(f"✓ Evaluation complete. Final results saved to: {out_json}")
    print("=" * 80)
    print(format_markdown_table(results_dict, tasks, master_task))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
