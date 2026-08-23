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
        "--export_adapter",
        action="store_true",
        default=False,
        help="Export merged weights as a standalone HuggingFace PEFT LoRA adapter.",
    )
    parser.add_argument(
        "--target_r",
        type=int,
        default=None,
        help="Optional target rank for exported PEFT adapter (e.g. 16 or 32).",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=32.0,
        help="LoRA alpha for exported PEFT adapter (default: 32.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to perform projections ('cpu' or 'cuda').",
    )
    args = parser.parse_args()
    if not args.config and not args.master:
        parser.error("Either --config or --master must be provided.")
    return args


def main():
    args = parse_args()

    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = MergeConfig.from_yaml(args.config)
    else:
        config = MergeConfig(
            master_adapter_path=args.master,
            plugin_adapter_paths=args.plugins,
            output_dir=args.output_dir,
            energy_threshold=args.energy_threshold,
            export_adapter=args.export_adapter,
            target_r=args.target_r,
            lora_alpha=args.lora_alpha,
            device=args.device,
        )

    print("\nInitializing Asymmetric NSP Merger...")
    print(f"  • Master:   {config.master_adapter_path}")
    print(f"  • Plugins:  {config.plugin_adapter_paths}")
    print(f"  • SVD Cutoff Threshold: {config.energy_threshold}")

    merger = AsymmetricNSPMerger(config)

    if args.export_adapter or config.export_adapter:
        print(f"\nExporting standalone PEFT adapter to {config.output_dir}...")
        merger.export_as_peft_adapter(
            output_dir=config.output_dir,
            target_r=config.target_r,
            lora_alpha=config.lora_alpha,
        )
    elif args.base_model:
        merged_deltas = merger.merge_adapters()
        print(f"\n✓ Successfully computed merged deltas for {len(merged_deltas)} modules.")
        print(f"\nApplying merged deltas to base model: {args.base_model}...")
        merger.apply_and_save(
            base_model_path=args.base_model,
            output_dir=config.output_dir,
            merged_deltas=merged_deltas,
        )
    else:
        merged_deltas = merger.merge_adapters()
        print(f"\n✓ Successfully computed merged deltas for {len(merged_deltas)} modules.")
        out_file = Path(config.output_dir) / "merged_deltas.pt"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save(merged_deltas, out_file)
        print(f"✓ Saved merged delta state dictionary to {out_file}")


if __name__ == "__main__":
    main()
