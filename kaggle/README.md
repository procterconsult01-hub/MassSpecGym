# Kaggle offline bundle (Enveda CASMI 2026)

Self-contained copy of `spec2smiles` + smoke config + predict/train scripts for
Featured Code Competition notebooks that cannot `pip install -e .` from GitHub.

## Layout

```
kaggle/
  spec2smiles/          # same modules as src/spec2smiles
  configs/enveda_cpu_smoke.yaml
  scripts/predict_kaggle.py
  scripts/train.py
```

## Use on Kaggle

1. Push this repo to GitHub (or zip `kaggle/` + `notebooks/`).
2. Create a Kaggle Dataset from the repo (or just the `kaggle/` folder) and attach it to the competition notebook.
3. Upload `artifacts/checkpoints/enveda_cpu_smoke.pt` as a separate Dataset (`.pt` files are gitignored).
4. In the notebook, add both Datasets + the competition data (`enveda-CASMI26-molecule-id-mass-spectra`).
5. Prefer the notebook `notebooks/enveda_casmi26_submit.ipynb`, which discovers paths under `/kaggle/input/`.

Keep `kaggle/spec2smiles` in sync with `src/spec2smiles` when you change the model.
