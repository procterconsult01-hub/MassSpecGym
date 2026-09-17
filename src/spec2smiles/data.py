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
                "formula": None,
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
    formula_col = cols.get("formula")
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
        formula = None
        if formula_col and pd.notna(r[formula_col]):
            formula = str(r[formula_col]).strip()
        rows.append(
            {
                "identifier": str(r[id_col]) if id_col else f"row_{len(rows)}",
                "mzs": mzs,
                "intensities": intensities,
                "smiles": smi,
                "formula": formula,
                "precursor_mz": float(r[prec_col]) if prec_col and pd.notna(r[prec_col]) else None,
                "fold": str(r[fold_col]) if fold_col else "train",
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows




def _default_enveda_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "kaggle" / "enveda-casmi26"


def load_enveda_parquet_rows(
    path: Path | str | None = None,
    *,
    split: str = "train",
    max_samples: int | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
    columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Load Enveda CASMI 2026 Kaggle parquet into MassSpecGym-style rows.

    Train: peaks from ms2_mzs / ms2_normalized_intensities; SMILES from
    normalized_smiles. No official fold — we carve a deterministic val split
    from the loaded subset (last val_fraction, or a held-out slice after
    max_samples for train+val).

    split:
      - "train" / "val": from train.parquet with internal split
      - "test": from test.parquet (smiles may be empty placeholder)
    """
    import pyarrow.parquet as pq

    root = _default_enveda_dir()
    split = (split or "train").lower()
    if path is None:
        path = root / ("test.parquet" if split == "test" else "train.parquet")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Enveda parquet not found: {path}")

    if split == "test":
        want_cols = columns or [
            "molecule_id",
            "spectrum_id",
            "ms2_mzs",
            "ms2_normalized_intensities",
            "precursor_mz",
            "adduct",
            "ionization_mode",
            "instrument_type",
            "molecular_formula",
        ]
        # test may lack molecular_formula
        schema_names = set(pq.read_schema(path).names)
        want_cols = [c for c in want_cols if c in schema_names]
        df = pq.read_table(path, columns=want_cols).to_pandas()
        rows: list[dict[str, Any]] = []
        for i, r in df.iterrows():
            mzs = _parse_array_cell(r.get("ms2_mzs"))
            intensities = _parse_array_cell(r.get("ms2_normalized_intensities"))
            if not mzs:
                continue
            if len(intensities) != len(mzs):
                n = min(len(mzs), len(intensities))
                mzs, intensities = mzs[:n], intensities[:n]
            mid = str(r.get("molecule_id", f"row_{i}"))
            sid = str(r.get("spectrum_id", mid))
            pmz = r.get("precursor_mz")
            rows.append(
                {
                    "identifier": sid,
                    "molecule_id": mid,
                    "spectrum_id": sid,
                    "mzs": mzs,
                    "intensities": intensities,
                    "smiles": "",  # unknown at test time
                    "formula": str(r["molecular_formula"]) if "molecular_formula" in r and pd.notna(r.get("molecular_formula")) else None,
                    "precursor_mz": float(pmz) if pmz is not None and pd.notna(pmz) else None,
                    "fold": "test",
                    "adduct": str(r["adduct"]) if "adduct" in r and pd.notna(r.get("adduct")) else None,
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
        return rows

    # train / val from train.parquet — stream row groups until enough rows
    want_cols = columns or [
        "normalized_smiles",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_mz",
        "molecular_formula",
        "inchikey",
    ]
    schema_names = set(pq.read_schema(path).names)
    want_cols = [c for c in want_cols if c in schema_names]

    # Need train+val pool when splitting
    if max_samples is not None:
        # load a bit more than requested so we can split
        if split in ("train", "val"):
            # Caller passes max_train / max_val separately; for a single call we
            # interpret max_samples as the size of THIS split. Upstream
            # build_dataloaders will call twice — so we load a shared pool
            # keyed by seed when both are needed. Here: load max_samples for
            # the requested split using a deterministic offset.
            pool_target = max_samples
        else:
            pool_target = max_samples
    else:
        pool_target = None

    pf = pq.ParquetFile(path)
    collected: list[dict[str, Any]] = []
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=want_cols)
        df = table.to_pandas()
        for _, r in df.iterrows():
            smi = r.get("normalized_smiles")
            if smi is None or (isinstance(smi, float) and math.isnan(smi)):
                continue
            smi = str(smi).strip()
            if not smi or smi.lower() == "nan":
                continue
            mzs = _parse_array_cell(r.get("ms2_mzs"))
            intensities = _parse_array_cell(r.get("ms2_normalized_intensities"))
            if not mzs:
                continue
            if len(intensities) != len(mzs):
                n = min(len(mzs), len(intensities))
                mzs, intensities = mzs[:n], intensities[:n]
            pmz = r.get("precursor_mz")
            formula = None
            if "molecular_formula" in r.index and pd.notna(r.get("molecular_formula")):
                formula = str(r["molecular_formula"]).strip()
            ik = None
            if "inchikey" in r.index and pd.notna(r.get("inchikey")):
                ik = str(r["inchikey"]).strip()
            collected.append(
                {
                    "identifier": ik or f"enveda_{len(collected)}",
                    "molecule_id": ik or f"enveda_{len(collected)}",
                    "mzs": mzs,
                    "intensities": intensities,
                    "smiles": smi,
                    "formula": formula,
                    "precursor_mz": float(pmz) if pmz is not None and pd.notna(pmz) else None,
                    "fold": "train",  # reassigned below
                    "inchikey": ik,
                }
            )
            if pool_target is not None and len(collected) >= pool_target:
                break
        if pool_target is not None and len(collected) >= pool_target:
            break

    if not collected:
        return []

    # Deterministic shuffle then split for train/val when loading full capped pool.
    # For separate train/val calls with different max_samples, use index ranges
    # from a shared larger pool. Simpler approach used by build_dataloaders:
    # load_enveda_train_val() — see below. For this function: assign folds by
    # hash so train and val calls with different max_samples still get disjoint
    # rows when reading sequentially from the start.
    rng = random.Random(seed)
    indices = list(range(len(collected)))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(collected) * val_fraction))) if len(collected) > 1 else 0
    val_set = set(indices[:n_val])
    out: list[dict[str, Any]] = []
    for i, row in enumerate(collected):
        row = dict(row)
        row["fold"] = "val" if i in val_set else "train"
        if row["fold"] == split:
            out.append(row)
    if max_samples is not None:
        out = out[:max_samples]
    return out


def load_enveda_train_val(
    data_dir: Path | str | None = None,
    max_train_samples: int | None = 5000,
    max_val_samples: int | None = 500,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a capped train/val split from Enveda train.parquet in one pass."""
    import pyarrow.parquet as pq

    data_dir = Path(data_dir) if data_dir else _default_enveda_dir()
    path = data_dir / "train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Enveda train parquet not found: {path}")

    n_train = int(max_train_samples) if max_train_samples is not None else 5000
    n_val = int(max_val_samples) if max_val_samples is not None else 500
    need = n_train + n_val

    want_cols = [
        "normalized_smiles",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_mz",
        "molecular_formula",
        "inchikey",
    ]
    schema_names = set(pq.read_schema(path).names)
    want_cols = [c for c in want_cols if c in schema_names]

    pf = pq.ParquetFile(path)
    collected: list[dict[str, Any]] = []
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=want_cols).to_pandas()
        for _, r in df.iterrows():
            smi = r.get("normalized_smiles")
            if smi is None or (isinstance(smi, float) and math.isnan(smi)):
                continue
            smi = str(smi).strip()
            if not smi or smi.lower() == "nan":
                continue
            mzs = _parse_array_cell(r.get("ms2_mzs"))
            intensities = _parse_array_cell(r.get("ms2_normalized_intensities"))
            if not mzs:
                continue
            if len(intensities) != len(mzs):
                n = min(len(mzs), len(intensities))
                mzs, intensities = mzs[:n], intensities[:n]
            pmz = r.get("precursor_mz")
            formula = None
            if "molecular_formula" in r.index and pd.notna(r.get("molecular_formula")):
                formula = str(r["molecular_formula"]).strip()
            ik = None
            if "inchikey" in r.index and pd.notna(r.get("inchikey")):
                ik = str(r["inchikey"]).strip()
            collected.append(
                {
                    "identifier": ik or f"enveda_{len(collected)}",
                    "molecule_id": ik or f"enveda_{len(collected)}",
                    "mzs": mzs,
                    "intensities": intensities,
                    "smiles": smi,
                    "formula": formula,
                    "precursor_mz": float(pmz) if pmz is not None and pd.notna(pmz) else None,
                    "fold": "train",
                    "inchikey": ik,
                }
            )
            if len(collected) >= need:
                break
        if len(collected) >= need:
            break

    rng = random.Random(seed)
    rng.shuffle(collected)
    val_rows = collected[:n_val]
    train_rows = collected[n_val : n_val + n_train]
    for r in val_rows:
        r["fold"] = "val"
    for r in train_rows:
        r["fold"] = "train"
    return train_rows, val_rows


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
    """Character-level SMILES or token-level SELFIES vocabulary."""

    def __init__(self, chars: list[str], token_level: bool = False):
        self.token_level = token_level
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

    def _tokenize(self, text: str) -> list[str]:
        if self.token_level:
            from spec2smiles.chem_utils import split_selfies_tokens

            return split_selfies_tokens(text)
        return list(text)

    def encode(self, smiles: str, max_len: int, add_special: bool = True) -> list[int]:
        ids: list[int] = []
        if add_special:
            ids.append(self.bos_id)
        for ch in self._tokenize(smiles):
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
        return {"itos": self.itos, "token_level": self.token_level}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SmilesVocab":
        itos = d["itos"]
        token_level = bool(d.get("token_level", False))
        chars = [t for t in itos if t not in SPECIAL_TOKENS]
        v = cls(chars, token_level=token_level)
        v.itos = list(itos)
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v


def build_smiles_vocab(smiles_list: list[str], decode_mode: str = "smiles") -> SmilesVocab:
    mode = (decode_mode or "smiles").lower()
    if mode == "selfies":
        from spec2smiles.chem_utils import split_selfies_tokens

        tokens: set[str] = set()
        for s in smiles_list:
            tokens.update(split_selfies_tokens(s))
        return SmilesVocab(sorted(tokens), token_level=True)
    chars: set[str] = set()
    for s in smiles_list:
        chars.update(s)
    return SmilesVocab(sorted(chars), token_level=False)


def prepare_targets(
    rows: list[dict[str, Any]],
    decode_mode: str = "smiles",
) -> list[dict[str, Any]]:
    """
    Attach training target string.
    For decode_mode=selfies: convert SMILES→SELFIES; drop rows that fail.
    Always keeps original smiles for metrics.
    """
    mode = (decode_mode or "smiles").lower()
    out: list[dict[str, Any]] = []
    if mode != "selfies":
        for r in rows:
            rr = dict(r)
            rr["target"] = r["smiles"]
            out.append(rr)
        return out
    from spec2smiles.chem_utils import smiles_to_selfies

    for r in rows:
        se = smiles_to_selfies(r["smiles"])
        if se is None:
            continue
        rr = dict(r)
        rr["target"] = se
        out.append(rr)
    return out


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
        target = r.get("target", r["smiles"])
        ids = self.vocab.encode(target, max_len=self.max_smiles_len)
        smi = torch.full((self.max_smiles_len,), self.vocab.pad_id, dtype=torch.long)
        n = min(len(ids), self.max_smiles_len)
        smi[:n] = torch.tensor(ids[:n], dtype=torch.long)
        smi_mask = smi == self.vocab.pad_id
        pmz = r.get("precursor_mz")
        precursor = torch.tensor(
            float(pmz) if pmz is not None else 0.0,
            dtype=torch.float32,
        )
        return {
            "mz_bin_ids": mz_bin_ids,
            "intensity": intensity,
            "mz_norm": mz_norm,
            "peak_mask": peak_mask,
            "smiles_ids": smi,
            "smiles_mask": smi_mask,
            "smiles": r["smiles"],
            "target": target,
            "precursor_mz": precursor,
            "identifier": r.get("identifier", str(idx)),
        }


def collate_batch(batch: list[dict]) -> dict[str, Any]:
    keys_t = [
        "mz_bin_ids",
        "intensity",
        "mz_norm",
        "peak_mask",
        "smiles_ids",
        "smiles_mask",
        "precursor_mz",
    ]
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys_t}
    out["smiles"] = [b["smiles"] for b in batch]
    out["target"] = [b.get("target", b["smiles"]) for b in batch]
    out["identifier"] = [b["identifier"] for b in batch]
    return out
