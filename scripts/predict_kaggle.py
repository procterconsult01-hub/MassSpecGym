#!/usr/bin/env python3
"""Predict Enveda CASMI 2026 Kaggle submission (25 SMILES per molecule_id)."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spec2smiles.chem_utils import is_valid_smiles  # noqa: E402
from spec2smiles.data import (  # noqa: E402
    SpectrumDataset,
    collate_batch,
    load_enveda_parquet_rows,
)
from spec2smiles.decode import beam_candidates_one  # noqa: E402
from spec2smiles.train import get_decode_mode, load_checkpoint, load_config, resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kaggle Enveda CASMI26 submission")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "artifacts" / "checkpoints" / "enveda_cpu_smoke.pt"),
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "enveda_cpu_smoke.yaml"),
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT / "data" / "kaggle" / "enveda-casmi26"),
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "artifacts" / "submissions" / "enveda_smoke.csv"),
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--beam-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--fill-smiles", type=str, default="CCO")
    p.add_argument("--max-spectra", type=int, default=None, help="Debug: limit test spectra")
    return p.parse_args()


def _rank_candidates(
    scored: list[tuple[float, str]],
    top_k: int,
    fill: str,
) -> list[str]:
    """Unique SMILES, prefer valid, best score first; pad to top_k with fill."""
    best: dict[str, float] = {}
    for sc, smi in scored:
        smi = (smi or "").strip()
        if not smi:
            continue
        if smi not in best or sc > best[smi]:
            best[smi] = sc
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    valid = [s for s, _ in ranked if is_valid_smiles(s)]
    invalid = [s for s, _ in ranked if not is_valid_smiles(s)]
    ordered = valid + invalid
    # dedupe preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for s in ordered:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    while len(uniq) < top_k:
        uniq.append(fill)
    return uniq[:top_k]


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model, vocab, ckpt_cfg = load_checkpoint(ckpt, device=device)
    # Prefer checkpoint data hyperparams for encoding consistency
    data_cfg = dict(cfg.get("data", {}))
    data_cfg.update({k: v for k, v in ckpt_cfg.get("data", {}).items() if k in (
        "max_peaks", "max_smiles_len", "mz_bins", "mz_max", "decode_mode",
    )})
    decode_mode = get_decode_mode(ckpt_cfg) if ckpt_cfg.get("data") else get_decode_mode(cfg)
    infer_cfg = {**cfg.get("infer", {}), **ckpt_cfg.get("infer", {})}
    beam_size = args.beam_size if args.beam_size is not None else int(infer_cfg.get("beam_size", 5))
    max_len = int(infer_cfg.get("max_len", data_cfg.get("max_smiles_len", 128)))
    top_k = int(args.top_k)
    fill = args.fill_smiles

    sample_path = data_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    molecule_order = sample["molecule_id"].astype(str).tolist()

    test_rows = load_enveda_parquet_rows(
        data_dir / "test.parquet",
        split="test",
        max_samples=args.max_spectra,
    )
    if not test_rows:
        raise RuntimeError("No test spectra loaded")

    # SpectrumDataset requires smiles key; use placeholder for encoding targets unused at infer
    for r in test_rows:
        if not r.get("smiles"):
            r["smiles"] = fill
            r["target"] = fill

    ds = SpectrumDataset(
        test_rows,
        vocab=vocab,
        max_peaks=int(data_cfg.get("max_peaks", 100)),
        max_smiles_len=int(data_cfg.get("max_smiles_len", 128)),
        mz_bins=int(data_cfg.get("mz_bins", 5000)),
        mz_max=float(data_cfg.get("mz_max", 2000.0)),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    # molecule_id -> list of (score, smiles)
    agg: dict[str, list[tuple[float, str]]] = defaultdict(list)
    # map dataset index -> molecule_id
    mol_ids = [str(r["molecule_id"]) for r in test_rows]

    model.eval()
    idx0 = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            B = batch["mz_bin_ids"].size(0)
            mz = batch["mz_bin_ids"].to(device)
            inten = batch["intensity"].to(device)
            mzn = batch["mz_norm"].to(device)
            pmask = batch["peak_mask"].to(device)
            pmz = batch.get("precursor_mz")
            if pmz is not None:
                pmz = pmz.to(device)
            for b in range(B):
                mid = mol_ids[idx0 + b]
                pmz_b = pmz[b : b + 1] if pmz is not None else None
                cands = beam_candidates_one(
                    model,
                    vocab,
                    mz[b : b + 1],
                    inten[b : b + 1],
                    mzn[b : b + 1],
                    pmask[b : b + 1],
                    beam_size=beam_size,
                    max_len=max_len,
                    precursor_mz=pmz_b,
                    decode_mode=decode_mode,
                )
                agg[mid].extend(cands)
            idx0 += B

    rows_out = []
    for mid in molecule_order:
        cands = _rank_candidates(agg.get(mid, []), top_k=top_k, fill=fill)
        rows_out.append({"molecule_id": mid, "smiles": ";".join(cands)})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame(rows_out, columns=["molecule_id", "smiles"])
    sub.to_csv(out_path, index=False)

    # Validate
    n_rows = len(sub)
    n_ok = 0
    n_unique_sets = 0
    for s in sub["smiles"]:
        parts = str(s).split(";")
        if len(parts) == top_k:
            n_ok += 1
        if len(set(parts)) > 1:
            n_unique_sets += 1
    covered = sum(1 for mid in molecule_order if mid in agg and agg[mid])
    print(f"Wrote {out_path}")
    print(f"rows={n_rows} (sample={len(molecule_order)}) exact_{top_k}_slots={n_ok}")
    print(f"molecules_with_preds={covered} rows_with_>1_unique={n_unique_sets}")
    print(f"spectra_scored={len(test_rows)} beam_size={beam_size} decode_mode={decode_mode}")


if __name__ == "__main__":
    main()
