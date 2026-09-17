"""Greedy / sampling decode for Spec2SmilesModel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from spec2smiles.data import SmilesVocab
    from spec2smiles.model import Spec2SmilesModel


@torch.no_grad()
def greedy_decode(
    model: "Spec2SmilesModel",
    vocab: "SmilesVocab",
    mz_bin_ids: torch.Tensor,
    intensity: torch.Tensor,
    mz_norm: torch.Tensor,
    peak_mask: torch.Tensor,
    max_len: int = 128,
    temperature: float = 1.0,
) -> list[str]:
    """
    Autoregressive greedy (or temperature) decode.
    Batch tensors: [B, P]
    """
    model.eval()
    device = mz_bin_ids.device
    B = mz_bin_ids.size(0)
    memory = model.encode(mz_bin_ids, intensity, mz_norm, peak_mask)

    ys = torch.full((B, 1), vocab.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_pad = ys == vocab.pad_id
        logits = model.decode(ys, memory, peak_mask, tgt_key_padding_mask=tgt_pad)
        next_logits = logits[:, -1, :] / max(temperature, 1e-5)
        if temperature <= 1e-5 or abs(temperature - 1.0) < 1e-6:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        # freeze finished sequences on EOS/PAD
        next_id = torch.where(
            finished.unsqueeze(1),
            torch.full_like(next_id, vocab.pad_id),
            next_id,
        )
        ys = torch.cat([ys, next_id], dim=1)
        finished = finished | (next_id.squeeze(1) == vocab.eos_id)
        if bool(finished.all()):
            break

    return [vocab.decode(ys[i]) for i in range(B)]


@torch.no_grad()
def decode_one(
    model: "Spec2SmilesModel",
    vocab: "SmilesVocab",
    sample: dict,
    device: torch.device,
    max_len: int = 128,
    temperature: float = 1.0,
) -> str:
    def _to(t):
        return t.unsqueeze(0).to(device)

    outs = greedy_decode(
        model,
        vocab,
        _to(sample["mz_bin_ids"]),
        _to(sample["intensity"]),
        _to(sample["mz_norm"]),
        _to(sample["peak_mask"]),
        max_len=max_len,
        temperature=temperature,
    )
    return outs[0]
