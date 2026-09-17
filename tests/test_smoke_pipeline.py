"""Minimal offline tests (no GPU required)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spec2smiles.data import (
    build_smiles_vocab,
    make_synthetic_rows,
    prepare_targets,
    SpectrumDataset,
    collate_batch,
)
from spec2smiles.model import Spec2SmilesModel
from spec2smiles.decode import greedy_decode, beam_search_decode, decode_batch
from spec2smiles.metrics import evaluate_predictions
from spec2smiles.chem_utils import prefer_valid_candidate
import torch


def test_forward_and_decode():
    rows = prepare_targets(make_synthetic_rows(16, seed=0), decode_mode="smiles")
    vocab = build_smiles_vocab([r["target"] for r in rows], decode_mode="smiles")
    ds = SpectrumDataset(rows, vocab, max_peaks=32, max_smiles_len=64, mz_bins=500)
    batch = collate_batch([ds[i] for i in range(4)])
    model = Spec2SmilesModel(
        vocab_size=len(vocab),
        mz_bins=500,
        d_model=32,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        pad_id=vocab.pad_id,
        max_smiles_len=64,
        mz_max=1000.0,
    )
    logits = model(
        batch["mz_bin_ids"],
        batch["intensity"],
        batch["mz_norm"],
        batch["peak_mask"],
        batch["smiles_ids"],
        batch["smiles_mask"],
        precursor_mz=batch["precursor_mz"],
    )
    assert logits.shape[0] == 4
    loss = model.loss(logits, batch["smiles_ids"])
    assert torch.isfinite(loss)
    preds = greedy_decode(
        model,
        vocab,
        batch["mz_bin_ids"],
        batch["intensity"],
        batch["mz_norm"],
        batch["peak_mask"],
        max_len=32,
        precursor_mz=batch["precursor_mz"],
    )
    assert len(preds) == 4
    beamed = beam_search_decode(
        model,
        vocab,
        batch["mz_bin_ids"],
        batch["intensity"],
        batch["mz_norm"],
        batch["peak_mask"],
        beam_size=3,
        max_len=24,
        precursor_mz=batch["precursor_mz"],
    )
    assert len(beamed) == 4
    m = evaluate_predictions(preds, batch["smiles"])
    assert m["n"] == 4


def test_prefer_valid_candidate():
    # invalid first (higher score) should lose to valid if prefer_valid
    best = prefer_valid_candidate(["((((", "CCO", "CC("], scores=[0.9, 0.5, 0.8])
    assert best == "CCO"
