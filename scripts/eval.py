#!/usr/bin/env python3
"""Evaluate a checkpoint: exact match, validity, optional Tanimoto."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torch.utils.data import DataLoader  # noqa: E402

from spec2smiles.data import SpectrumDataset, collate_batch, load_rows, prepare_targets  # noqa: E402
from spec2smiles.train import (  # noqa: E402
    ensure_fixture,
    evaluate_loader,
    get_decode_mode,
    load_checkpoint,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Spec2Smiles checkpoint")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "artifacts" / "checkpoints" / "smoke_spec2smiles_best.pt"),
    )
    p.add_argument("--source", type=str, default="synthetic")
    p.add_argument("--tsv", type=str, default=None)
    p.add_argument("--fold", type=str, default="val")
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--beam-size", type=int, default=None, help="Override infer.beam_size")
    p.add_argument("--out", type=str, default=None, help="Optional JSON metrics path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_fixture(ROOT)
    device = resolve_device(args.device)
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model, vocab, cfg = load_checkpoint(ckpt, device=device)
    data_cfg = cfg.get("data", {})
    decode_mode = get_decode_mode(cfg)
    infer_cfg = cfg.get("infer", {})
    beam_size = args.beam_size if args.beam_size is not None else int(infer_cfg.get("beam_size", 5))
    rows = load_rows(
        source=args.source,
        tsv_path=args.tsv or data_cfg.get("tsv_path"),
        fold=args.fold if args.source != "synthetic" else args.fold,
        max_samples=args.max_samples,
    )
    if args.source == "synthetic" and not rows:
        rows = load_rows(source="synthetic", fold=None, max_samples=args.max_samples)
    rows = prepare_targets(rows, decode_mode=decode_mode)

    ds = SpectrumDataset(
        rows,
        vocab=vocab,
        max_peaks=data_cfg.get("max_peaks", 100),
        max_smiles_len=data_cfg.get("max_smiles_len", 128),
        mz_bins=data_cfg.get("mz_bins", 5000),
        mz_max=data_cfg.get("mz_max", 1000.0),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    metrics = evaluate_loader(
        model,
        loader,
        vocab,
        device,
        max_len=int(infer_cfg.get("max_len", 128)),
        beam_size=beam_size,
        decode_mode=decode_mode,
        prefer_valid=bool(infer_cfg.get("prefer_valid", True)),
    )
    printable = {k: v for k, v in metrics.items() if k != "examples"}
    printable["examples"] = [{"true": t, "pred": p} for t, p in metrics.get("examples", [])]
    print(json.dumps(printable, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(printable, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
