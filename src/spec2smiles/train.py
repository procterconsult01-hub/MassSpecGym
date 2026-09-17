"""Training helpers for Spec2Smiles."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from spec2smiles.data import (
    SpectrumDataset,
    build_smiles_vocab,
    collate_batch,
    load_rows,
    prepare_targets,
    write_synthetic_fixture,
    SmilesVocab,
)
from spec2smiles.decode import decode_batch
from spec2smiles.metrics import evaluate_predictions
from spec2smiles.model import Spec2SmilesModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_config(path: str | Path | None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    cfg_path = Path(path) if path else root / "configs" / "default.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_smoke_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    smoke = cfg.get("smoke", {})
    cfg = dict(cfg)
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["model"] = dict(cfg.get("model", {}))
    cfg["train"] = dict(cfg.get("train", {}))
    cfg["data"]["source"] = "synthetic"
    cfg["data"]["max_train_samples"] = smoke.get("max_train_samples", 32)
    cfg["data"]["max_val_samples"] = smoke.get("max_val_samples", 8)
    cfg["train"]["epochs"] = smoke.get("epochs", 2)
    cfg["train"]["max_steps"] = smoke.get("max_steps", 20)
    cfg["train"]["batch_size"] = smoke.get("batch_size", 8)
    for k in ("d_model", "nhead", "num_encoder_layers", "num_decoder_layers", "dim_feedforward"):
        if k in smoke:
            cfg["model"][k] = smoke[k]
    return cfg


def ensure_fixture(root: Path) -> Path:
    path = root / "data" / "fixtures" / "synthetic.jsonl"
    if not path.exists():
        write_synthetic_fixture(path, n=64, seed=42)
    return path


def get_decode_mode(cfg: dict[str, Any]) -> str:
    return str(cfg.get("data", {}).get("decode_mode", "smiles")).lower()


def build_dataloaders(cfg: dict[str, Any], vocab: SmilesVocab | None = None):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    source = data_cfg.get("source", "auto")
    decode_mode = get_decode_mode(cfg)

    train_rows = load_rows(
        source=source,
        tsv_path=data_cfg.get("tsv_path"),
        hf_repo=data_cfg.get("hf_repo", "roman-bushuiev/MassSpecGym"),
        hf_file=data_cfg.get("hf_file", "data/MassSpecGym1.5.tsv"),
        fold=data_cfg.get("fold_train", "train"),
        max_samples=data_cfg.get("max_train_samples"),
    )
    val_rows = load_rows(
        source=source,
        tsv_path=data_cfg.get("tsv_path"),
        hf_repo=data_cfg.get("hf_repo", "roman-bushuiev/MassSpecGym"),
        hf_file=data_cfg.get("hf_file", "data/MassSpecGym1.5.tsv"),
        fold=data_cfg.get("fold_val", "val"),
        max_samples=data_cfg.get("max_val_samples"),
    )
    train_rows = prepare_targets(train_rows, decode_mode=decode_mode)
    val_rows = prepare_targets(val_rows, decode_mode=decode_mode)
    if not val_rows:
        val_rows = train_rows[: max(1, min(8, len(train_rows) // 5))]

    if vocab is None:
        vocab = build_smiles_vocab(
            [r["target"] for r in train_rows],
            decode_mode=decode_mode,
        )

    ds_kwargs = dict(
        vocab=vocab,
        max_peaks=data_cfg.get("max_peaks", 100),
        max_smiles_len=data_cfg.get("max_smiles_len", 128),
        mz_bins=data_cfg.get("mz_bins", 5000),
        mz_max=data_cfg.get("mz_max", 1000.0),
    )
    train_ds = SpectrumDataset(train_rows, **ds_kwargs)
    val_ds = SpectrumDataset(val_rows, **ds_kwargs)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
        collate_fn=collate_batch,
    )
    return train_loader, val_loader, vocab, train_rows, val_rows


def build_model(cfg: dict[str, Any], vocab: SmilesVocab) -> Spec2SmilesModel:
    m = cfg["model"]
    d = cfg["data"]
    return Spec2SmilesModel(
        vocab_size=len(vocab),
        mz_bins=d.get("mz_bins", 5000),
        d_model=m.get("d_model", 128),
        nhead=m.get("nhead", 4),
        num_encoder_layers=m.get("num_encoder_layers", 2),
        num_decoder_layers=m.get("num_decoder_layers", 2),
        dim_feedforward=m.get("dim_feedforward", 256),
        dropout=m.get("dropout", 0.1),
        peak_embed_dim=m.get("peak_embed_dim", 32),
        max_smiles_len=d.get("max_smiles_len", 128),
        pad_id=vocab.pad_id,
        use_precursor_mass=bool(m.get("use_precursor_mass", True)),
        mz_max=float(d.get("mz_max", 1000.0)),
    )


def save_checkpoint(
    path: Path,
    model: Spec2SmilesModel,
    vocab: SmilesVocab,
    cfg: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "vocab": vocab.to_dict(),
        "config": cfg,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path | str,
    device: torch.device | None = None,
) -> tuple[Spec2SmilesModel, SmilesVocab, dict[str, Any]]:
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    vocab = SmilesVocab.from_dict(payload["vocab"])
    cfg = payload["config"]
    model = build_model(cfg, vocab)
    # Allow loading older checkpoints missing mass_proj
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    if missing:
        print(f"checkpoint missing keys (ok if upgraded): {missing}")
    if unexpected:
        print(f"checkpoint unexpected keys: {unexpected}")
    model.to(device)
    model.eval()
    return model, vocab, cfg


@torch.no_grad()
def evaluate_loader(
    model: Spec2SmilesModel,
    loader: DataLoader,
    vocab: SmilesVocab,
    device: torch.device,
    max_batches: int | None = None,
    max_len: int = 128,
    beam_size: int = 1,
    decode_mode: str = "smiles",
    prefer_valid: bool = True,
) -> dict[str, Any]:
    model.eval()
    preds: list[str] = []
    trues: list[str] = []
    total_loss = 0.0
    n_tok = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        mz = batch["mz_bin_ids"].to(device)
        inten = batch["intensity"].to(device)
        mzn = batch["mz_norm"].to(device)
        pmask = batch["peak_mask"].to(device)
        smi = batch["smiles_ids"].to(device)
        smask = batch["smiles_mask"].to(device)
        pmz = batch.get("precursor_mz")
        if pmz is not None:
            pmz = pmz.to(device)
        logits = model(mz, inten, mzn, pmask, smi, smask, precursor_mz=pmz)
        loss = model.loss(logits, smi)
        total_loss += float(loss.item()) * smi.size(0)
        n_tok += smi.size(0)
        decoded = decode_batch(
            model,
            vocab,
            mz,
            inten,
            mzn,
            pmask,
            max_len=max_len,
            beam_size=beam_size,
            precursor_mz=pmz,
            prefer_valid=prefer_valid,
            decode_mode=decode_mode,
        )
        preds.extend(decoded)
        trues.extend(batch["smiles"])
    metrics = evaluate_predictions(preds, trues)
    metrics["loss"] = total_loss / max(n_tok, 1)
    metrics["examples"] = list(zip(trues[:5], preds[:5]))
    metrics["beam_size"] = beam_size
    metrics["decode_mode"] = decode_mode
    return metrics


def train_loop(cfg: dict[str, Any], smoke: bool = False) -> Path:
    root = Path(__file__).resolve().parents[2]
    if smoke:
        cfg = apply_smoke_overrides(cfg)
        ensure_fixture(root)

    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))
    decode_mode = get_decode_mode(cfg)
    train_loader, val_loader, vocab, _, _ = build_dataloaders(cfg)
    model = build_model(cfg, vocab).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 3e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )

    epochs = int(cfg["train"].get("epochs", 10))
    max_steps = cfg["train"].get("max_steps")
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    log_every = int(cfg["train"].get("log_every", 10))
    ckpt_dir = Path(cfg["train"].get("checkpoint_dir", "artifacts/checkpoints"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir
    ckpt_name = cfg["train"].get("checkpoint_name", "spec2smiles_best.pt")
    ckpt_path = ckpt_dir / ("smoke_" + ckpt_name if smoke else ckpt_name)

    infer_cfg = cfg.get("infer", {})
    beam_size = int(infer_cfg.get("beam_size", 1))
    # During epoch val, use smaller beam for speed unless smoke/longer explicitly wants it
    val_beam = int(infer_cfg.get("val_beam_size", min(beam_size, 3 if not smoke else 1)))
    max_len = int(infer_cfg.get("max_len", 128))

    print(
        f"device={device} params={model.count_parameters():,} vocab={len(vocab)} "
        f"decode_mode={decode_mode} train_batches={len(train_loader)} "
        f"beam={beam_size} val_beam={val_beam} smoke={smoke}"
    )

    global_step = 0
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        running = 0.0
        for step, batch in enumerate(pbar, start=1):
            mz = batch["mz_bin_ids"].to(device)
            inten = batch["intensity"].to(device)
            mzn = batch["mz_norm"].to(device)
            pmask = batch["peak_mask"].to(device)
            smi = batch["smiles_ids"].to(device)
            smask = batch["smiles_mask"].to(device)
            pmz = batch.get("precursor_mz")
            if pmz is not None:
                pmz = pmz.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(mz, inten, mzn, pmask, smi, smask, precursor_mz=pmz)
            loss = model.loss(logits, smi)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            global_step += 1
            running += float(loss.item())
            if step % log_every == 0 or step == 1:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if max_steps is not None and global_step >= int(max_steps):
                break

        val_metrics = evaluate_loader(
            model,
            val_loader,
            vocab,
            device,
            max_batches=5 if smoke else None,
            max_len=max_len,
            beam_size=val_beam,
            decode_mode=decode_mode,
        )
        row = {
            "epoch": epoch,
            "train_loss": running / max(step, 1),
            "val_loss": val_metrics["loss"],
            "exact_match": val_metrics["exact_match"],
            "validity": val_metrics["validity"],
            "tanimoto_mean": val_metrics.get("tanimoto_mean"),
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss≈{row['train_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"exact={val_metrics['exact_match']:.3f} "
            f"validity={val_metrics['validity']}"
        )
        if val_metrics["loss"] <= best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                ckpt_path,
                model,
                vocab,
                cfg,
                extra={
                    "epoch": epoch,
                    "val": {k: v for k, v in val_metrics.items() if k != "examples"},
                    "global_step": global_step,
                    "history": history,
                },
            )
            print(f"saved checkpoint → {ckpt_path}")

        if max_steps is not None and global_step >= int(max_steps):
            break

    if not ckpt_path.exists():
        save_checkpoint(
            ckpt_path,
            model,
            vocab,
            cfg,
            extra={"epoch": epochs, "global_step": global_step, "history": history},
        )
        print(f"saved checkpoint → {ckpt_path}")

    return ckpt_path
