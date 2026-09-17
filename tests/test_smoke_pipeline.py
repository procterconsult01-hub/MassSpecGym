"""Minimal offline tests (no GPU required)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spec2smiles.data import build_smiles_vocab, make_synthetic_rows, SpectrumDataset, collate_batch
from spec2smiles.model import Spec2SmilesModel
from spec2smiles.decode import greedy_decode
from spec2smiles.metrics import evaluate_predictions
import torch


def test_forward_and_decode():
    rows = make_synthetic_rows(16, seed=0)
    vocab = build_smiles_vocab([r["smiles"] for r in rows])
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
    )
    logits = model(
        batch["mz_bin_ids"],
        batch["intensity"],
        batch["mz_norm"],
        batch["peak_mask"],
        batch["smiles_ids"],
        batch["smiles_mask"],
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
    )
    assert len(preds) == 4
    m = evaluate_predictions(preds, batch["smiles"])
    assert m["n"] == 4
