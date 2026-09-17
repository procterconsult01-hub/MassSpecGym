"""Greedy / beam decode for Spec2SmilesModel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from spec2smiles.chem_utils import prefer_valid_candidate, selfies_to_smiles

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
    precursor_mz: Optional[torch.Tensor] = None,
) -> list[str]:
    """
    Autoregressive greedy (or temperature) decode.
    Batch tensors: [B, P]
    """
    model.eval()
    device = mz_bin_ids.device
    B = mz_bin_ids.size(0)
    memory = model.encode(mz_bin_ids, intensity, mz_norm, peak_mask, precursor_mz=precursor_mz)

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
def beam_search_decode(
    model: "Spec2SmilesModel",
    vocab: "SmilesVocab",
    mz_bin_ids: torch.Tensor,
    intensity: torch.Tensor,
    mz_norm: torch.Tensor,
    peak_mask: torch.Tensor,
    beam_size: int = 5,
    max_len: int = 128,
    precursor_mz: Optional[torch.Tensor] = None,
    prefer_valid: bool = True,
    decode_mode: str = "smiles",
) -> list[str]:
    """
    Batch beam search. Returns one string per batch item (best / preferred-valid).
    When beam_size <= 1, falls back to greedy.
    For decode_mode=selfies, returns SMILES (after conversion + validity prefer).
    """
    if beam_size is None or beam_size <= 1:
        return greedy_decode(
            model, vocab, mz_bin_ids, intensity, mz_norm, peak_mask,
            max_len=max_len, precursor_mz=precursor_mz,
        )

    model.eval()
    device = mz_bin_ids.device
    B = mz_bin_ids.size(0)
    # Encode once; expand memory for beams per sample sequentially for clarity/CPU.
    # Process one spectrum at a time for simpler beam bookkeeping on CPU.
    outs: list[str] = []
    for b in range(B):
        pmz = precursor_mz[b : b + 1] if precursor_mz is not None else None
        outs.append(
            _beam_one(
                model,
                vocab,
                mz_bin_ids[b : b + 1],
                intensity[b : b + 1],
                mz_norm[b : b + 1],
                peak_mask[b : b + 1],
                beam_size=beam_size,
                max_len=max_len,
                precursor_mz=pmz,
                prefer_valid=prefer_valid,
                decode_mode=decode_mode,
            )
        )
    return outs


def _beam_one(
    model: "Spec2SmilesModel",
    vocab: "SmilesVocab",
    mz_bin_ids: torch.Tensor,
    intensity: torch.Tensor,
    mz_norm: torch.Tensor,
    peak_mask: torch.Tensor,
    beam_size: int,
    max_len: int,
    precursor_mz: Optional[torch.Tensor],
    prefer_valid: bool,
    decode_mode: str = "smiles",
) -> str:
    device = mz_bin_ids.device
    memory = model.encode(mz_bin_ids, intensity, mz_norm, peak_mask, precursor_mz=precursor_mz)
    # beams: list of (score, token_ids tensor [1, T], finished)
    bos = torch.tensor([[vocab.bos_id]], dtype=torch.long, device=device)
    beams: list[tuple[float, torch.Tensor, bool]] = [(0.0, bos, False)]
    completed: list[tuple[float, torch.Tensor]] = []

    for _ in range(max_len - 1):
        cand_next: list[tuple[float, torch.Tensor, bool]] = []
        active = [(s, y, f) for s, y, f in beams if not f]
        if not active:
            break
        # batch active beams
        ys = torch.cat([y for _, y, _ in active], dim=0)
        scores = [s for s, _, _ in active]
        mem = memory.expand(ys.size(0), -1, -1)
        pmask = peak_mask.expand(ys.size(0), -1)
        tgt_pad = ys == vocab.pad_id
        logits = model.decode(ys, mem, pmask, tgt_key_padding_mask=tgt_pad)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # [A, V]
        topk = min(beam_size, log_probs.size(-1))
        vals, idxs = log_probs.topk(topk, dim=-1)
        for a in range(ys.size(0)):
            for k in range(topk):
                tok = int(idxs[a, k].item())
                new_score = scores[a] + float(vals[a, k].item())
                new_y = torch.cat(
                    [ys[a : a + 1], torch.tensor([[tok]], device=device, dtype=torch.long)],
                    dim=1,
                )
                finished = tok == vocab.eos_id
                if finished:
                    completed.append((new_score, new_y))
                else:
                    cand_next.append((new_score, new_y, False))

        # keep top beam_size active
        cand_next.sort(key=lambda x: x[0], reverse=True)
        beams = cand_next[:beam_size]
        if not beams and completed:
            break

    if not completed:
        # take unfinished beams as completed
        completed = [(s, y) for s, y, _ in beams]

    # length-normalized scores
    scored: list[tuple[float, str]] = []
    for score, y in completed:
        text = vocab.decode(y[0])
        # normalize by length (exclude bos)
        length = max(y.size(1) - 1, 1)
        scored.append((score / length, text))

    # unique by string, keep best score
    best_by_text: dict[str, float] = {}
    for sc, text in scored:
        if text not in best_by_text or sc > best_by_text[text]:
            best_by_text[text] = sc
    candidates = list(best_by_text.keys())
    scores_list = [best_by_text[c] for c in candidates]
    # If training in SELFIES space, convert candidates to SMILES before RDKit filter
    if decode_mode == "selfies":
        converted: list[str] = []
        conv_scores: list[float] = []
        for c, sc in zip(candidates, scores_list):
            smi = selfies_to_smiles(c)
            converted.append(smi if smi is not None else c)
            conv_scores.append(sc)
        candidates, scores_list = converted, conv_scores
    if prefer_valid:
        return prefer_valid_candidate(candidates, scores_list)
    best_i = max(range(len(candidates)), key=lambda i: scores_list[i])
    return candidates[best_i]


def postprocess_prediction(text: str, decode_mode: str = "smiles") -> str:
    """Convert model output string to SMILES for metrics / display."""
    mode = (decode_mode or "smiles").lower()
    if mode == "selfies":
        smi = selfies_to_smiles(text)
        return smi if smi is not None else text
    return text


@torch.no_grad()
def decode_batch(
    model: "Spec2SmilesModel",
    vocab: "SmilesVocab",
    mz_bin_ids: torch.Tensor,
    intensity: torch.Tensor,
    mz_norm: torch.Tensor,
    peak_mask: torch.Tensor,
    max_len: int = 128,
    beam_size: int = 1,
    temperature: float = 1.0,
    precursor_mz: Optional[torch.Tensor] = None,
    prefer_valid: bool = True,
    decode_mode: str = "smiles",
) -> list[str]:
    if beam_size and beam_size > 1:
        # beam_search already converts SELFIES→SMILES when prefer_valid/decode_mode set
        return beam_search_decode(
            model, vocab, mz_bin_ids, intensity, mz_norm, peak_mask,
            beam_size=beam_size, max_len=max_len, precursor_mz=precursor_mz,
            prefer_valid=prefer_valid,
            decode_mode=decode_mode,
        )
    raw = greedy_decode(
        model, vocab, mz_bin_ids, intensity, mz_norm, peak_mask,
        max_len=max_len, temperature=temperature, precursor_mz=precursor_mz,
    )
    return [postprocess_prediction(t, decode_mode=decode_mode) for t in raw]


@torch.no_grad()
def decode_one(
    model: "Spec2SmilesModel",
    vocab: "SmilesVocab",
    sample: dict,
    device: torch.device,
    max_len: int = 128,
    temperature: float = 1.0,
    beam_size: int = 1,
    prefer_valid: bool = True,
    decode_mode: str = "smiles",
) -> str:
    def _to(t):
        return t.unsqueeze(0).to(device)

    pmz = sample.get("precursor_mz")
    pmz_t = None
    if pmz is not None and torch.is_tensor(pmz):
        pmz_t = _to(pmz)
    elif pmz is not None:
        pmz_t = torch.tensor([float(pmz)], dtype=torch.float32, device=device)

    outs = decode_batch(
        model,
        vocab,
        _to(sample["mz_bin_ids"]),
        _to(sample["intensity"]),
        _to(sample["mz_norm"]),
        _to(sample["peak_mask"]),
        max_len=max_len,
        beam_size=beam_size,
        temperature=temperature,
        precursor_mz=pmz_t,
        prefer_valid=prefer_valid,
        decode_mode=decode_mode,
    )
    return outs[0]
