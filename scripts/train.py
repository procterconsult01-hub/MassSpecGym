#!/usr/bin/env python3
"""Train Spec2Smiles on MassSpecGym (or synthetic smoke data)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install: add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spec2smiles.train import load_config, train_loop  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train spectrum → SMILES model")
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--smoke", action="store_true", help="Tiny offline smoke run")
    p.add_argument("--source", type=str, default=None, help="synthetic|hf|tsv|auto")
    p.add_argument("--tsv", type=str, default=None, help="Path to MassSpecGym TSV")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap training rows (subset / CPU runs)",
    )
    p.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Cap validation rows (subset / CPU runs)",
    )
    p.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Override train.checkpoint_name",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.source:
        cfg["data"]["source"] = args.source
    if args.tsv:
        cfg["data"]["tsv_path"] = args.tsv
        cfg["data"]["source"] = "tsv"
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.device:
        cfg["device"] = args.device
    if args.max_train_samples is not None:
        cfg["data"]["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        cfg["data"]["max_val_samples"] = args.max_val_samples
    if args.checkpoint_name:
        cfg["train"]["checkpoint_name"] = args.checkpoint_name
    ckpt = train_loop(cfg, smoke=args.smoke)
    print(f"DONE checkpoint={ckpt}")


if __name__ == "__main__":
    main()
