#!/usr/bin/env python3
"""Run inference: load checkpoint, print SMILES for a sample spectrum."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spec2smiles.data import SpectrumDataset, load_rows, prepare_targets  # noqa: E402
from spec2smiles.decode import decode_one  # noqa: E402
from spec2smiles.train import (  # noqa: E402
    ensure_fixture,
    get_decode_mode,
    load_checkpoint,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer SMILES from a spectrum")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "artifacts" / "checkpoints" / "smoke_spec2smiles_best.pt"),
    )
    p.add_argument("--index", type=int, default=0, help="Sample index from dataset")
    p.add_argument("--source", type=str, default="synthetic")
    p.add_argument("--tsv", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beam-size", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_fixture(ROOT)
    device = resolve_device(args.device)
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        alt = ROOT / "artifacts" / "checkpoints" / "spec2smiles_best.pt"
        if alt.exists():
            ckpt = alt
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}. Run scripts/train.py --smoke first."
            )

    model, vocab, cfg = load_checkpoint(ckpt, device=device)
    data_cfg = cfg.get("data", {})
    decode_mode = get_decode_mode(cfg)
    infer_cfg = cfg.get("infer", {})
    beam_size = args.beam_size if args.beam_size is not None else int(infer_cfg.get("beam_size", 1))
    rows = load_rows(
        source=args.source,
        tsv_path=args.tsv or data_cfg.get("tsv_path"),
        fold=None,
        max_samples=max(args.index + 1, 8),
    )
    if not rows:
        raise RuntimeError("No spectrum rows loaded")
    rows = prepare_targets(rows, decode_mode=decode_mode)
    idx = args.index % len(rows)
    ds = SpectrumDataset(
        rows,
        vocab=vocab,
        max_peaks=data_cfg.get("max_peaks", 100),
        max_smiles_len=data_cfg.get("max_smiles_len", 128),
        mz_bins=data_cfg.get("mz_bins", 5000),
        mz_max=data_cfg.get("mz_max", 1000.0),
    )
    sample = ds[idx]
    pred = decode_one(
        model,
        vocab,
        sample,
        device=device,
        max_len=args.max_len,
        temperature=args.temperature,
        beam_size=beam_size,
        prefer_valid=bool(infer_cfg.get("prefer_valid", True)),
        decode_mode=decode_mode,
    )
    true = sample["smiles"]
    print(f"identifier: {sample['identifier']}")
    print(f"true_smiles: {true}")
    print(f"pred_smiles: {pred}")
    print(f"beam_size: {beam_size} decode_mode: {decode_mode}")


if __name__ == "__main__":
    main()
