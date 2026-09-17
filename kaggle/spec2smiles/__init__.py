"""spec2smiles: Transformer baseline for MS/MS → SMILES (MassSpecGym de novo)."""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Spec2SmilesModel",
    "SpectrumDataset",
    "build_smiles_vocab",
]


def __getattr__(name: str):
    if name == "Spec2SmilesModel":
        from spec2smiles.model import Spec2SmilesModel

        return Spec2SmilesModel
    if name == "SpectrumDataset":
        from spec2smiles.data import SpectrumDataset

        return SpectrumDataset
    if name == "build_smiles_vocab":
        from spec2smiles.data import build_smiles_vocab

        return build_smiles_vocab
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
