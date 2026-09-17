# CPU longer run — MassSpecGym spectra → SMILES (SELFIES + beam)

**When:** Wed Sep 16, 2026 ~11:50 PM – 12:00 AM CDT (box UTC Sep 17 04:50–04:58)  
**Device:** CPU only (no GPU)  
**Checkpoint:** `artifacts/checkpoints/spec2smiles_cpu_longer.pt` (~5.4 MB; best by val loss @ epoch 1)

## What changed vs prior 2k subset

| Feature | Prior (`cpu_subset`) | This run (`cpu_longer`) |
|---------|----------------------|-------------------------|
| Train / val size | 2k / 400 | **10k / 1k** |
| Epochs | 3 | **5** |
| Decode space | SMILES chars | **SELFIES tokens** (`decode_mode: selfies`) |
| Inference | greedy (`beam_size: 1`) | **beam 5** + RDKit prefer-valid |
| Conditioning | peaks only | + **precursor_mz** mass embedding |
| Post-hoc | none | SELFIES round-trip repair on invalid beams |

## Data / config

- File: `data/MassSpecGym1.5.tsv` (~231k rows)
- Config: `configs/cpu_longer.yaml`
- Model: Transformer enc–dec, d_model=128, 2+2 layers, ~**1.34M** params
- Vocab: **47** SELFIES tokens (from 10k train subset)
- Train: batch_size=16, lr=3e-4, ~625 steps/epoch
- Epoch val used `val_beam_size: 1` (speed); final eval used **beam_size: 5**

## Train / val loss (epoch val, greedy decode)

| Epoch | train_loss (approx) | val_loss | exact_match | validity (RDKit) |
|------:|--------------------:|---------:|------------:|-----------------:|
| 1 | 1.5620 | **2.1623** (best → saved) | 0.000 | **1.0** |
| 2 | 0.9452 | 2.3732 | 0.000 | 1.0 |
| 3 | 0.7022 | 2.5193 | 0.000 | 1.0 |
| 4 | 0.5509 | 2.6535 | 0.000 | 1.0 |
| 5 | 0.4529 | 2.8127 | 0.000 | 1.0 |

Train loss fell; val loss rose after epoch 1 (overfit on 10k subset — same pattern as the 2k run). Full log: `artifacts/logs/cpu_longer_train.log`.

> Note: absolute val_loss is not directly comparable to the SMILES-char 2k run (different token space / vocab).

## Post-train eval (held-out val, n=64, beam=5)

Saved JSON: `artifacts/reports/cpu_longer_eval.json`

### Comparison to prior 2k SMILES run

| Metric | Prior 2k / 3ep (greedy SMILES) | This 10k / 5ep (beam5 SELFIES) |
|--------|-------------------------------:|-------------------------------:|
| loss (eval) | 1.474 | 1.778 |
| exact_match | 0.0 | 0.0 |
| **validity** | **0.0** | **1.0** |
| **tanimoto_mean** | null (no valid preds) | **0.138** |
| rdkit_available | true | true |

**Validity lift:** 0 → 1.0 on the 64-sample eval set (SELFIES grammar + beam + RDKit prefer-valid). Exact match remains 0 at this scale; Tanimoto is now measurable (~0.14 mean when both valid).

## Example inference

```
identifier: MassSpecGymID0000001
true_smiles: COc1cc([C@H](Cc2ccccc2)NC(C)=O)oc(=O)c1
pred_smiles: CCCCCC/C=C\C/C=C\CCCCCCCCC(=O)OCCCCCCCCCCCCCCCCCCC   # valid, wrong molecule
beam_size: 5 decode_mode: selfies
```

## How to reproduce

```bash
cd /workspace/MassSpecGym
source .venv/bin/activate
pip install selfies rdkit   # if needed

# Train longer CPU subset
python scripts/train.py --config configs/cpu_longer.yaml

# Eval with beam 5
python scripts/eval.py --checkpoint artifacts/checkpoints/spec2smiles_cpu_longer.pt \
  --source tsv --tsv data/MassSpecGym1.5.tsv --fold val --max-samples 64 --device cpu \
  --beam-size 5 --out artifacts/reports/cpu_longer_eval.json

# Single-spectrum infer
python scripts/infer.py --checkpoint artifacts/checkpoints/spec2smiles_cpu_longer.pt \
  --source tsv --tsv data/MassSpecGym1.5.tsv --device cpu --index 0 --beam-size 5
```

## Code landed

- `src/spec2smiles/decode.py` — real **beam search**; RDKit prefer-valid; SELFIES→SMILES postprocess
- `src/spec2smiles/chem_utils.py` — RDKit / SELFIES helpers + repair
- `src/spec2smiles/data.py` — `decode_mode`, SELFIES targets/vocab, `precursor_mz` in batches, formula column passthrough
- `src/spec2smiles/model.py` — optional precursor mass embedding
- `src/spec2smiles/train.py` — wires beam / decode_mode into train & eval
- `configs/cpu_longer.yaml` — 10k/1k/5ep/beam5/selfies
- Optional dep: `selfies` in `pyproject.toml` / `requirements.txt`

## Notes

- No GitHub push (no token). Local commit only.
- Exact-match competitive MassSpecGym scores still need much more data / capacity / GPU.
- Best checkpoint is epoch 1 by val CE; validity stays ~1.0 across epochs thanks to SELFIES.
