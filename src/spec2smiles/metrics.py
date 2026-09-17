"""Evaluation metrics for de novo SMILES prediction."""

from __future__ import annotations

from typing import Any


def exact_match(pred: str, true: str) -> bool:
    return pred.strip() == true.strip()


def smiles_validity(smiles: str) -> bool | None:
    """
    RDKit validity. Returns None if RDKit is unavailable.
    """
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None
    if not smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def tanimoto_similarity(pred: str, true: str, radius: int = 2, n_bits: int = 2048) -> float | None:
    """ECFP Tanimoto; None if RDKit missing or either SMILES invalid."""
    try:
        from rdkit import Chem
        from rdkit import DataStructs
        from rdkit.Chem import AllChem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None
    m1 = Chem.MolFromSmiles(pred)
    m2 = Chem.MolFromSmiles(true)
    if m1 is None or m2 is None:
        return None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius, nBits=n_bits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, radius, nBits=n_bits)
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def evaluate_predictions(
    preds: list[str],
    trues: list[str],
) -> dict[str, Any]:
    assert len(preds) == len(trues)
    n = len(preds)
    if n == 0:
        return {"n": 0, "exact_match": 0.0, "validity": None, "tanimoto_mean": None}

    em = sum(1 for p, t in zip(preds, trues) if exact_match(p, t)) / n

    valids = [smiles_validity(p) for p in preds]
    if all(v is None for v in valids):
        validity: float | None = None
        rdkit_available = False
    else:
        rdkit_available = True
        # treat None as False if mixed (shouldn't happen)
        validity = sum(1 for v in valids if v) / n

    tans = [tanimoto_similarity(p, t) for p, t in zip(preds, trues)]
    tans_f = [x for x in tans if x is not None]
    tan_mean = sum(tans_f) / len(tans_f) if tans_f else None

    return {
        "n": n,
        "exact_match": em,
        "validity": validity,
        "tanimoto_mean": tan_mean,
        "rdkit_available": rdkit_available,
    }
