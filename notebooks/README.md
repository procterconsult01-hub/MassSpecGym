# Kaggle notebooks — Enveda CASMI 2026

Competition: [`enveda-CASMI26-molecule-id-mass-spectra`](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra) (Featured **Code Competition**).

| Notebook | Purpose |
|----------|---------|
| `enveda_casmi26_submit.ipynb` | Install deps → resolve `/kaggle/input` paths → load checkpoint **or** short train → write `/kaggle/working/submission.csv` |

Related repo pieces:

- `scripts/predict_kaggle.py` — local CLI predict (same ranking / 25 SMILES format)
- `scripts/train.py --config configs/enveda_cpu_smoke.yaml` — smoke / subset train
- `kaggle/` — offline copy of `spec2smiles` + config + scripts for attaching as a Dataset
- Checkpoint (gitignored): `artifacts/checkpoints/enveda_cpu_smoke.pt`

## Upload & submit steps

### 1. Push this repo to GitHub

```bash
cd MassSpecGym
# ensure notebooks/*.ipynb and kaggle/ are committed (no tokens, no parquet)
git status
git push -u origin main
```

Do **not** commit `~/.kaggle/kaggle.json`, `.env`, or `data/kaggle/**/*.parquet`.

### 2. Add code as a Kaggle Dataset (or clone in-notebook)

Options:

- **Dataset from GitHub** — Kaggle → Datasets → New → link this repo; attach to the competition notebook.
- **Zip upload** — zip the repo (or just `kaggle/` + `notebooks/` + `configs/` + `src/`) and upload as a Dataset.
- The notebook prefers `src/spec2smiles` when the full repo is attached; otherwise it uses the `kaggle/spec2smiles` bundle.

### 3. Upload the smoke checkpoint as a Dataset

`*.pt` files are gitignored. After local smoke train:

```bash
python scripts/train.py --config configs/enveda_cpu_smoke.yaml
# → artifacts/checkpoints/enveda_cpu_smoke.pt
```

Create a Kaggle Dataset (e.g. `enveda-smoke-ckpt`) containing that file and attach it.

Alternatively set `RUN_MODE = "train_subset"` in the notebook to train briefly on GPU/CPU inside the Code Comp run (slower; honest first baseline is still checkpoint).

### 4. Create competition notebook → Save Version → Submit

1. Open the competition → **Code** → **New Notebook**.
2. **Add data**: competition input + code Dataset + checkpoint Dataset.
3. Upload / copy `notebooks/enveda_casmi26_submit.ipynb` (or File → Import).
4. Confirm path cell finds:
   - competition dir with `sample_submission.csv` + `test.parquet`
   - `spec2smiles` on `sys.path`
   - a `.pt` checkpoint (unless `train_subset`)
5. **Save Version** → **Save & Run All** (required for Code Competitions).
6. When green, **Submit** the version output `submission.csv`.

Expected submission schema:

```text
molecule_id,smiles
m_....,SMILES1;SMILES2;...;SMILES25
```

Exactly **25** semicolon-separated candidates per row; row order must match `sample_submission.csv`.

### 5. Local dry-run (optional)

```bash
source .venv/bin/activate
python scripts/predict_kaggle.py \
  --checkpoint artifacts/checkpoints/enveda_cpu_smoke.pt \
  --config configs/enveda_cpu_smoke.yaml \
  --data-dir data/kaggle/enveda-casmi26 \
  --output artifacts/submissions/enveda_smoke.csv
```

You can also open `enveda_casmi26_submit.ipynb` locally; it falls back to `/workspace/MassSpecGym` (or the repo root) when `/kaggle/input` is absent.
