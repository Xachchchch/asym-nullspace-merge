"""02_run_merge.py: CLI entry point to merge real LoRA adapters using Asymmetric NSP."""

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asym_nsp.merger import AsymmetricNSPMerger, MergeConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters with Two-Sided Asymmetric Null-Space Projection."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (e.g. configs/merge_reasoning_code.yaml).",
    )
    parser.add_argument(
        "--master",
        type=str,
        default=None,
        help="Master LoRA adapter directory or HuggingFace repo ID.",
    )
    parser.add_argument(
        "--plugins",
        type=str,
        nargs="+",
        default=[],
        help="One or more Plugin LoRA adapter directories or HuggingFace repo IDs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./merged_output",
        help="Destination directory for merged weights.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Optional base model path to apply merged delta weights directly.",
    )
    parser.add_argument(
        "--energy_threshold",
        type=float,
        default=0.99,
        help="Cumulative SVD energy threshold (default: 0.99).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to perform projections ('cpu' or 'cuda').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = MergeConfig.from_yaml(args.config)
    else:
        if not args.master:
            parser.error("Either --config or --master must be provided.")
        config = MergeConfig(
            master_adapter_path=args.master,
            plugin_adapter_paths=args.plugins,
            output_dir=args.output_dir,
            energy_threshold=args.energy_threshold,
            device=args.device,
        )

    print("\nInitializing Asymmetric NSP Merger...")
    print(f"  • Master:   {config.master_adapter_path}")
    print(f"  • Plugins:  {config.plugin_adapter_paths}")
    print(f"  • SVD Cutoff Threshold: {config.energy_threshold}")

    merger = AsymmetricNSPMerger(config)
    merged_deltas = merger.merge_adapters()
    print(f"\n✓ Successfully computed merged deltas for {len(merged_deltas)} modules.")

    if args.base_model:
        print(f"\nApplying merged deltas to base model: {args.base_model}...")
        merger.apply_and_save(
            base_model_path=args.base_model,
            output_dir=config.output_dir,
            merged_deltas=merged_deltas,
        )
    else:
        out_file = Path(config.output_dir) / "merged_deltas.pt"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save(merged_deltas, out_file)
        print(f"✓ Saved merged delta state dictionary to {out_file}")


if __name__ == "__main__":
    main()
