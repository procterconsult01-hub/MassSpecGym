"""Dataset loaders for MassSpecGym spectra → SMILES."""

from __future__ import annotations

import ast
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Special tokens for SMILES char vocab
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# Tiny synthetic SMILES for offline smoke tests
SYNTHETIC_SMILES = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CCN",
    "C1CCCCC1",
    "CC(C)O",
    "CC(=O)C",
    "COC",
    "c1ccncc1",
    "CCCl",
    "CCC",
    "CCCC",
    "CCOC",
    "CCN(C)C",
    "c1ccc(O)cc1",
    "CC(=O)N",
    "CCS",
    "CC#N",
    "CCOCC",
    "c1ccccc1O",
    "CC(C)C",
    "CCC(=O)O",
    "c1ccc(N)cc1",
    "CCBr",
    "CCOC(=O)C",
    "CC(O)C",
    "c1ccoc1",
    "CCCCC",
    "CC(=O)OC",
    "c1ccc(C)cc1",
    "CCNC",
    "CO",
]


def _parse_array_cell(cell: Any) -> list[float]:
    """Parse mzs/intensities cell from TSV (string list or already list)."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [float(x) for x in cell]
    s = str(cell).strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [float(x) for x in val]
    except (ValueError, SyntaxError):
        pass
    # comma / space separated
    parts = s.replace("[", "").replace("]", "").replace(";", ",").split(",")
    out: list[float] = []
    for p in parts:
        p = p.strip()
        if p:
            try:
                out.append(float(p))
            except ValueError:
                continue
    return out


def make_synthetic_rows(n: int = 64, seed: int = 42) -> list[dict[str, Any]]:
    """Generate synthetic spectrum→SMILES pairs for offline smoke tests."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        smi = SYNTHETIC_SMILES[i % len(SYNTHETIC_SMILES)]
        # Deterministic-ish fake peaks from hash of SMILES + index
        n_peaks = rng.randint(5, 25)
        mzs = sorted(rng.uniform(50.0, 400.0) for _ in range(n_peaks))
        # Bias a few peaks by character codes so model can learn something weak
        for j, ch in enumerate(smi[: min(5, len(smi))]):
            if j < len(mzs):
                mzs[j] = 50.0 + (ord(ch) % 50) + j * 17.3
        mzs = sorted(mzs)
        intensities = [rng.uniform(0.05, 1.0) for _ in mzs]
        mx = max(intensities) if intensities else 1.0
        intensities = [x / mx for x in intensities]
        fold = "train" if i < int(0.8 * n) else ("val" if i < int(0.9 * n) else "test")
        rows.append(
            {
                "identifier": f"synth_{i:04d}",
                "mzs": mzs,
                "intensities": intensities,
                "smiles": smi,
                "precursor_mz": float(mzs[-1]) + 1.0 if mzs else 100.0,
                "fold": fold,
            }
        )
    return rows


def write_synthetic_fixture(path: Path | str, n: int = 64, seed: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = make_synthetic_rows(n=n, seed=seed)
    # JSONL for easy load without pandas edge cases
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def load_synthetic_fixture(path: Path | str | None = None) -> list[dict[str, Any]]:
    if path is None:
        default = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "synthetic.jsonl"
        path = default
    path = Path(path)
    if not path.exists():
        write_synthetic_fixture(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def try_load_massspecgym_package(fold: str | None = None) -> list[dict[str, Any]] | None:
    """Attempt to use official massspecgym package if installed."""
    try:
        import massspecgym  # noqa: F401
    except ImportError:
        return None
    # Package APIs vary by version; prefer HF TSV path if unclear
    return None


def download_hf_tsv(
    repo: str = "roman-bushuiev/MassSpecGym",
    filename: str = "data/MassSpecGym1.5.tsv",
    dest_dir: Path | str = "data",
) -> Path:
    """Download MassSpecGym TSV from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    return Path(local)


def load_tsv_rows(
    path: Path | str,
    fold: str | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    path = Path(path)
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    mz_col = cols.get("mzs", cols.get("mz"))
    int_col = cols.get("intensities", cols.get("intensity"))
    smi_col = cols.get("smiles")
    fold_col = cols.get("fold")
    id_col = cols.get("identifier", cols.get("id"))
    prec_col = cols.get("precursor_mz")
    if mz_col is None or int_col is None or smi_col is None:
        raise ValueError(
            f"TSV missing required columns. Found: {list(df.columns)}. "
            "Need mzs, intensities, smiles."
        )
    if fold and fold_col:
        df = df[df[fold_col].astype(str).str.lower() == fold.lower()]
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        mzs = _parse_array_cell(r[mz_col])
        intensities = _parse_array_cell(r[int_col])
        smi = str(r[smi_col]).strip()
        if not mzs or not smi or smi.lower() == "nan":
            continue
        if len(intensities) != len(mzs):
            n = min(len(mzs), len(intensities))
            mzs, intensities = mzs[:n], intensities[:n]
        rows.append(
            {
                "identifier": str(r[id_col]) if id_col else f"row_{len(rows)}",
                "mzs": mzs,
                "intensities": intensities,
                "smiles": smi,
                "precursor_mz": float(r[prec_col]) if prec_col and pd.notna(r[prec_col]) else None,
                "fold": str(r[fold_col]) if fold_col else "train",
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def load_rows(
    source: str = "auto",
    tsv_path: str | Path | None = None,
    hf_repo: str = "roman-bushuiev/MassSpecGym",
    hf_file: str = "data/MassSpecGym1.5.tsv",
    fold: str | None = None,
    max_samples: int | None = None,
    synthetic_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Load spectrum rows.

    Priority for source=auto:
      1. Explicit tsv_path if present
      2. data/MassSpecGym1.5.tsv or data/MassSpecGym.tsv locally
      3. Try HF download (may fail offline)
      4. Synthetic fixture
    """
    source = (source or "auto").lower()
    root = Path(__file__).resolve().parents[2]

    if source == "synthetic":
        rows = load_synthetic_fixture(synthetic_path)
        if fold:
            rows = [r for r in rows if r.get("fold", "train") == fold]
        if max_samples is not None:
            rows = rows[:max_samples]
        return rows

    if source in ("tsv", "hf", "auto", "massspecgym"):
        candidates: list[Path] = []
        if tsv_path:
            candidates.append(Path(tsv_path))
        candidates.extend(
            [
                root / "data" / "MassSpecGym1.5.tsv",
                root / "data" / "MassSpecGym.tsv",
                root / "data" / "data" / "MassSpecGym1.5.tsv",
            ]
        )
        for p in candidates:
            if p.exists():
                return load_tsv_rows(p, fold=fold, max_samples=max_samples)

        if source in ("hf", "auto"):
            try:
                local = download_hf_tsv(repo=hf_repo, filename=hf_file, dest_dir=root / "data")
                return load_tsv_rows(local, fold=fold, max_samples=max_samples)
            except Exception as exc:  # noqa: BLE001
                if source == "hf":
                    raise RuntimeError(f"HF download failed: {exc}") from exc
                # fall through to synthetic for auto

        if source == "massspecgym":
            pkg_rows = try_load_massspecgym_package(fold=fold)
            if pkg_rows is not None:
                if max_samples is not None:
                    pkg_rows = pkg_rows[:max_samples]
                return pkg_rows

    # Fallback
    rows = load_synthetic_fixture(synthetic_path)
    if fold:
        rows = [r for r in rows if r.get("fold", "train") == fold]
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


class SmilesVocab:
    """Character-level SMILES vocabulary."""

    def __init__(self, chars: list[str]):
        self.itos = list(SPECIAL_TOKENS) + sorted(set(chars))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, smiles: str, max_len: int, add_special: bool = True) -> list[int]:
        ids: list[int] = []
        if add_special:
            ids.append(self.bos_id)
        for ch in smiles:
            ids.append(self.stoi.get(ch, self.unk_id))
            if max_len and len(ids) >= max_len - (1 if add_special else 0):
                break
        if add_special:
            ids.append(self.eos_id)
        if max_len and len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = self.eos_id
        return ids

    def decode(self, ids: list[int] | torch.Tensor, skip_special: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        special = {self.pad_id, self.bos_id, self.eos_id}
        chars: list[str] = []
        for i in ids:
            if i == self.eos_id:
                break
            if skip_special and i in special:
                continue
            if 0 <= i < len(self.itos):
                tok = self.itos[i]
                if tok not in SPECIAL_TOKENS:
                    chars.append(tok)
        return "".join(chars)

    def to_dict(self) -> dict[str, Any]:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SmilesVocab":
        itos = d["itos"]
        # Rebuild without duplicating specials already in itos
        chars = [t for t in itos if t not in SPECIAL_TOKENS]
        v = cls(chars)
        # Preserve exact itos order from checkpoint
        v.itos = list(itos)
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v


def build_smiles_vocab(smiles_list: list[str]) -> SmilesVocab:
    chars: set[str] = set()
    for s in smiles_list:
        chars.update(s)
    return SmilesVocab(sorted(chars))


def peaks_to_tensors(
    mzs: list[float],
    intensities: list[float],
    max_peaks: int,
    mz_bins: int,
    mz_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      mz_bin_ids: LongTensor [max_peaks]
      intensity: FloatTensor [max_peaks]
      mz_norm: FloatTensor [max_peaks]  (mz / mz_max)
      peak_mask: BoolTensor [max_peaks] True = padding
    """
    n = min(len(mzs), max_peaks)
    mz_bin_ids = torch.zeros(max_peaks, dtype=torch.long)
    intensity = torch.zeros(max_peaks, dtype=torch.float32)
    mz_norm = torch.zeros(max_peaks, dtype=torch.float32)
    peak_mask = torch.ones(max_peaks, dtype=torch.bool)
    if n == 0:
        return mz_bin_ids, intensity, mz_norm, peak_mask
    # sort by intensity descending, keep top max_peaks
    order = sorted(range(len(mzs)), key=lambda i: intensities[i], reverse=True)[:n]
    order = sorted(order, key=lambda i: mzs[i])  # re-sort by m/z for locality
    for j, i in enumerate(order):
        mz = float(mzs[i])
        inten = float(intensities[i])
        b = int(min(max(mz, 0.0), mz_max) / mz_max * (mz_bins - 1))
        mz_bin_ids[j] = b
        intensity[j] = inten
        mz_norm[j] = min(mz, mz_max) / mz_max
        peak_mask[j] = False
    # renormalize intensities among kept peaks
    s = float(intensity[:n].sum())
    if s > 0:
        intensity[:n] = intensity[:n] / s
    return mz_bin_ids, intensity, mz_norm, peak_mask


class SpectrumDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        vocab: SmilesVocab,
        max_peaks: int = 100,
        max_smiles_len: int = 128,
        mz_bins: int = 5000,
        mz_max: float = 1000.0,
    ):
        self.rows = rows
        self.vocab = vocab
        self.max_peaks = max_peaks
        self.max_smiles_len = max_smiles_len
        self.mz_bins = mz_bins
        self.mz_max = mz_max

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        r = self.rows[idx]
        mz_bin_ids, intensity, mz_norm, peak_mask = peaks_to_tensors(
            r["mzs"], r["intensities"], self.max_peaks, self.mz_bins, self.mz_max
        )
        ids = self.vocab.encode(r["smiles"], max_len=self.max_smiles_len)
        # pad
        smi = torch.full((self.max_smiles_len,), self.vocab.pad_id, dtype=torch.long)
        n = min(len(ids), self.max_smiles_len)
        smi[:n] = torch.tensor(ids[:n], dtype=torch.long)
        smi_mask = smi == self.vocab.pad_id
        return {
            "mz_bin_ids": mz_bin_ids,
            "intensity": intensity,
            "mz_norm": mz_norm,
            "peak_mask": peak_mask,
            "smiles_ids": smi,
            "smiles_mask": smi_mask,
            "smiles": r["smiles"],
            "identifier": r.get("identifier", str(idx)),
        }


def collate_batch(batch: list[dict]) -> dict[str, Any]:
    keys_t = ["mz_bin_ids", "intensity", "mz_norm", "peak_mask", "smiles_ids", "smiles_mask"]
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys_t}
    out["smiles"] = [b["smiles"] for b in batch]
    out["identifier"] = [b["identifier"] for b in batch]
    return out
