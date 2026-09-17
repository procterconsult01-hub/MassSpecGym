"""Chemistry helpers: RDKit validity, SELFIES convert/repair."""

from __future__ import annotations

from typing import Optional


def rdkit_available() -> bool:
    try:
        from rdkit import Chem  # noqa: F401

        return True
    except ImportError:
        return False


def selfies_available() -> bool:
    try:
        import selfies  # noqa: F401

        return True
    except ImportError:
        return False


def is_valid_smiles(smiles: str) -> bool:
    if not smiles:
        return False
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:  # noqa: BLE001
        return False


def smiles_to_selfies(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    try:
        import selfies as sf
    except ImportError:
        return None
    try:
        return sf.encoder(smiles)
    except Exception:  # noqa: BLE001
        return None


def selfies_to_smiles(selfies_str: str) -> Optional[str]:
    if not selfies_str:
        return None
    try:
        import selfies as sf
    except ImportError:
        return None
    try:
        smi = sf.decoder(selfies_str)
        return smi if smi else None
    except Exception:  # noqa: BLE001
        return None


def split_selfies_tokens(selfies_str: str) -> list[str]:
    import selfies as sf

    return list(sf.split_selfies(selfies_str))


def selfies_roundtrip_repair(smiles: str) -> Optional[str]:
    """
    If SMILES is (nearly) encodable, round-trip through SELFIES.
    Returns a SMILES string or None.
    """
    se = smiles_to_selfies(smiles)
    if se is None:
        # Sometimes broken SMILES still decode if treated as SELFIES-ish; skip.
        return None
    out = selfies_to_smiles(se)
    if out and is_valid_smiles(out):
        return out
    return out


def prefer_valid_candidate(candidates: list[str], scores: list[float] | None = None) -> str:
    """
    Prefer chemically valid (RDKit) candidates; among those, highest score
    (scores are log-probs, higher better). Falls back to best score / first.
    Applies optional SELFIES round-trip repair when still invalid.
    """
    if not candidates:
        return ""
    if scores is None:
        scores = [0.0] * len(candidates)
    ranked = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)

    valid_idx: list[int] = []
    repaired: dict[int, str] = {}
    for i in ranked:
        c = candidates[i]
        if is_valid_smiles(c):
            valid_idx.append(i)
        else:
            fixed = selfies_roundtrip_repair(c)
            if fixed and is_valid_smiles(fixed):
                repaired[i] = fixed
                valid_idx.append(i)

    if valid_idx:
        best = valid_idx[0]
        return repaired.get(best, candidates[best])

    # No valid: still try repair on top-ranked
    for i in ranked:
        fixed = selfies_roundtrip_repair(candidates[i])
        if fixed:
            return fixed
    return candidates[ranked[0]]
