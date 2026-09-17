# CPU subset run — MassSpecGym spectra → SMILES

**When:** Wed Sep 16, 2026 ~11:37–11:41 PM CDT  
**Device:** CPU only (no GPU)  
**Checkpoint:** `artifacts/checkpoints/spec2smiles_cpu_subset.pt` (5.2 MB; best by val loss @ epoch 1)

## Data

| Item | Value |
|------|-------|
| File | `data/MassSpecGym1.5.tsv` |
| Source | Hugging Face `roman-bushuiev/MassSpecGym` → `data/MassSpecGym1.5.tsv` |
| Size | **250 MB** (261,197,365 bytes) |
| Rows | **231,104** (train 194,119 / val 19,429 / test 17,556) |
| Subset | max_train_samples=**2000**, max_val_samples=**400** |

## Config

- Config file: `configs/cpu_subset.yaml`
- Model: Transformer enc–dec, d_model=128, 2+2 layers, ~**1.33M** params, vocab size **32** (chars from subset)
- Train: epochs=**3**, batch_size=**16**, lr=3e-4, max_steps=null (~125 steps/epoch)

## Train / val loss

| Epoch | train_loss (approx) | val_loss | exact_match | validity (RDKit) |
|------:|--------------------:|---------:|------------:|-----------------:|
| 1 | 2.0354 | **1.6973** (best → saved) | 0.000 | 0.0 |
| 2 | 1.3046 | 1.7420 | 0.000 | 0.0 |
| 3 | 1.0910 | 1.7445 | 0.000 | 0.0225 |

Train loss fell across epochs; val loss rose slightly after epoch 1 (expected for a tiny 2k-sample CPU run). Full log: `artifacts/logs/cpu_subset_train.log`.

## Post-train eval (held-out val, n=64)

Saved JSON: `artifacts/reports/cpu_subset_eval.json`

| Metric | Value |
|--------|------:|
| loss | 1.474 |
| exact_match | 0.0 |
| validity | 0.0 |
| tanimoto_mean | null (no valid preds) |
| rdkit_available | true |

Exact-match / validity near zero is expected: 3 epochs on 2k samples is a smoke-scale baseline, not a competitive MassSpecGym de novo result. Predictions look like generic carbon-chain / carbonyl patterns with unbalanced parentheses.

## Example inferences (first TSV rows, indices 0–2)

| id | true (trunc) | pred (trunc) |
|----|--------------|--------------|
| MassSpecGymID0000001 | `COc1cc([C@H](Cc2ccccc2)NC(C)=O)oc(=O)c1` | `CCCC(=O)C(=O)OC(C)C)...` (invalid) |
| MassSpecGymID0000002 | same scaffold family | similar invalid chain |
| MassSpecGymID0000003 | same | similar |

## How to reproduce

```bash
cd /workspace/MassSpecGym
source .venv/bin/activate

# 1) Data (skip if data/MassSpecGym1.5.tsv already present)
python -c "from pathlib import Path; from spec2smiles.data import download_hf_tsv; p=download_hf_tsv(dest_dir=Path('data')); print(p)"
# If HF nests under data/data/, move: mv data/data/MassSpecGym1.5.tsv data/

# Optional chemistry metrics
pip install rdkit

# 2) Train subset (CPU)
python scripts/train.py --config configs/cpu_subset.yaml
# Equivalent CLI overrides:
# python scripts/train.py --config configs/default.yaml --source tsv \
#   --tsv data/MassSpecGym1.5.tsv --device cpu \
#   --max-train-samples 2000 --max-val-samples 400 \
#   --epochs 3 --batch-size 16 \
#   --checkpoint-name spec2smiles_cpu_subset.pt

# 3) Infer / eval
python scripts/infer.py --checkpoint artifacts/checkpoints/spec2smiles_cpu_subset.pt \
  --source tsv --tsv data/MassSpecGym1.5.tsv --device cpu --index 0
python scripts/eval.py --checkpoint artifacts/checkpoints/spec2smiles_cpu_subset.pt \
  --source tsv --tsv data/MassSpecGym1.5.tsv --fold val --max-samples 64 --device cpu \
  --out artifacts/reports/cpu_subset_eval.json
```

## Notes

- No GitHub push in this run (no token). Code changes: `configs/cpu_subset.yaml`, CLI flags `--max-train-samples` / `--max-val-samples` / `--checkpoint-name` on `scripts/train.py`.
- RDKit installed in `.venv` for validity / Tanimoto when predictions are valid.
- Larger data / more epochs / GPU would be needed for non-trivial exact-match rates.
