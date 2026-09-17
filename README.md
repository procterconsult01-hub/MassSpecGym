# MassSpecGym — Spec2Smiles baseline

Local Python baseline for **de novo molecular structure generation**: predict a **SMILES** string from an **LC-MS/MS** spectrum (peak list of *m/z* and intensity).

This project targets the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) de novo benchmark and the Hugging Face dataset [`roman-bushuiev/MassSpecGym`](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym). It is a **small Transformer** (~few M params in smoke mode) intended as a runnable starting point, not a SOTA method.

> **Note:** Files are created under `/workspace/MassSpecGym` only. Nothing is pushed to the private user repo.

## Problem

Given a tandem MS/MS spectrum \((m/z_i, I_i)_{i=1}^{n}\) (and optional precursor metadata), generate the corresponding molecular SMILES. MassSpecGym defines train/val/test folds with structure-aware splits to reduce leakage.

**Model:** peak tokens (discretized *m/z* + intensity features) → Transformer encoder → autoregressive char-level SMILES decoder.

## Setup

```bash
cd /workspace/MassSpecGym   # or your checkout path
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
# Lighter CPU-only torch (optional; default pip may pull CUDA wheels ~several GB):
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# then: pip install -e . --no-deps && pip install numpy pandas pyyaml tqdm huggingface_hub
# Optional chemistry metrics:
# pip install rdkit
# Optional official Lightning package (heavy; not required — we have HF TSV fallback):
# pip install "massspecgym @ git+https://github.com/pluskal-lab/MassSpecGym.git"
```

Minimal deps are listed in `requirements.txt` / `pyproject.toml` (`torch`, `numpy`, `pandas`, `pyyaml`, `tqdm`, `huggingface_hub`).

### Data sources (automatic fallback)

| Priority | Source | How |
|----------|--------|-----|
| 1 | Local TSV | Place `data/MassSpecGym1.5.tsv` (or set `--tsv`) |
| 2 | Hugging Face | `data.source: hf` downloads `data/MassSpecGym1.5.tsv` from `roman-bushuiev/MassSpecGym` |
| 3 | Synthetic | Offline fixture `data/fixtures/synthetic.jsonl` (used by `--smoke`) |

Manual HF download:

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('roman-bushuiev/MassSpecGym','data/MassSpecGym1.5.tsv',repo_type='dataset',local_dir='data'))"
```

Or:

```python
from datasets import load_dataset
ds = load_dataset("roman-bushuiev/MassSpecGym", data_files="data/MassSpecGym1.5.tsv")
```

## Smoke train (offline)

Uses synthetic spectrum→SMILES pairs, 1–2 epochs / few steps, small model:

```bash
source .venv/bin/activate
python scripts/train.py --smoke
```

Checkpoint: `artifacts/checkpoints/smoke_spec2smiles_best.pt`

## Full train

With a local TSV:

```bash
python scripts/train.py --source tsv --tsv data/MassSpecGym1.5.tsv --epochs 10 --batch-size 32
```

Or auto/HF (needs network):

```bash
python scripts/train.py --source hf --epochs 10
```

Config defaults: `configs/default.yaml`. Checkpoint: `artifacts/checkpoints/spec2smiles_best.pt`

## Inference

```bash
python scripts/infer.py --checkpoint artifacts/checkpoints/smoke_spec2smiles_best.pt --index 0
```

Prints `true_smiles` and `pred_smiles` for one sample.

## Evaluation

```bash
python scripts/eval.py --checkpoint artifacts/checkpoints/smoke_spec2smiles_best.pt --source synthetic
```

Metrics:

- **Exact match** — string equality of predicted vs reference SMILES
- **Validity** — RDKit `MolFromSmiles` success rate (if `rdkit` installed)
- **Tanimoto** — mean ECFP4 similarity when RDKit is available

## Layout

```
MassSpecGym/
  configs/default.yaml
  data/fixtures/synthetic.jsonl   # created on first smoke
  scripts/train.py
  scripts/infer.py
  scripts/eval.py
  src/spec2smiles/
    __init__.py
    data.py      # loaders, vocab, SpectrumDataset
    model.py     # Transformer encoder–decoder
    train.py     # train loop helpers
    decode.py    # greedy decode
    metrics.py   # exact / validity / Tanimoto
  artifacts/checkpoints/
  pyproject.toml
  requirements.txt
```



## Enveda CASMI 2026 (Kaggle)

Competition: [`enveda-CASMI26-molecule-id-mass-spectra`](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)

Local data (already downloaded):

```
data/kaggle/enveda-casmi26/
  train.parquet      # ~2.54M spectra; target = normalized_smiles
  test.parquet       # 1213 spectra, 400 unique molecule_id
  sample_submission.csv  # molecule_id, smiles (25 candidates joined by ';')
```

Peaks map to the existing spectrum encoding (`ms2_mzs` → `mzs`, `ms2_normalized_intensities` → `intensities`). Multiple test spectra can share one `molecule_id`; the submission must have **one row per molecule_id** with exactly **25** semicolon-separated SMILES.

### Smoke train (CPU, small subset)

```bash
source .venv/bin/activate
python scripts/train.py --config configs/enveda_cpu_smoke.yaml
```

Config defaults: 5k train / 500 val, 2 epochs (capped at 200 steps), `decode_mode: selfies`, checkpoint `artifacts/checkpoints/enveda_cpu_smoke.pt`.

### Kaggle Code Competition notebook

See `notebooks/enveda_casmi26_submit.ipynb` and `notebooks/README.md` for Featured Code Competition upload/submit steps (attach competition data + `kaggle/` code Dataset + smoke checkpoint Dataset → Save Version → Submit). Offline modules live under `kaggle/spec2smiles`.

### Kaggle predict / submission

```bash
python scripts/predict_kaggle.py \
  --checkpoint artifacts/checkpoints/enveda_cpu_smoke.pt \
  --config configs/enveda_cpu_smoke.yaml \
  --output artifacts/submissions/enveda_smoke.csv
```

The script beam-decodes every test spectrum, aggregates candidates per `molecule_id`, keeps top-25 unique (prefer RDKit-valid), pads with `CCO`, and writes a CSV matching `sample_submission` columns/order.

## Citations

```bibtex
@inproceedings{bushuiev2024massspecgym,
  author = {Bushuiev, Roman and Bushuiev, Anton and Samusevich, Raman and others},
  title = {MassSpecGym: A benchmark for the discovery and identification of molecules},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2024},
  url = {https://github.com/pluskal-lab/MassSpecGym}
}
```

Dataset: [roman-bushuiev/MassSpecGym](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym) · Paper: [arXiv:2410.23326](https://arxiv.org/abs/2410.23326)

## License

MIT for this baseline scaffolding. Respect MassSpecGym / dataset licenses when using the full TSV.
