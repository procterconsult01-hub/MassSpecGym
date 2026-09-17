"""Small Transformer: spectrum peak encoder → autoregressive SMILES/SELFIES decoder."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeakEmbedding(nn.Module):
    """Embed discrete m/z bins + continuous intensity / mz features."""

    def __init__(self, mz_bins: int, d_model: int, peak_embed_dim: int = 32):
        super().__init__()
        self.mz_embed = nn.Embedding(mz_bins, d_model)
        self.cont_proj = nn.Linear(2, peak_embed_dim)
        self.fuse = nn.Linear(d_model + peak_embed_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        mz_bin_ids: torch.Tensor,
        intensity: torch.Tensor,
        mz_norm: torch.Tensor,
    ) -> torch.Tensor:
        e = self.mz_embed(mz_bin_ids)
        cont = self.cont_proj(torch.stack([intensity, mz_norm], dim=-1))
        x = self.fuse(torch.cat([e, cont], dim=-1))
        return self.norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Spec2SmilesModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        mz_bins: int = 5000,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        peak_embed_dim: int = 32,
        max_smiles_len: int = 128,
        pad_id: int = 0,
        use_precursor_mass: bool = True,
        mz_max: float = 1000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.max_smiles_len = max_smiles_len
        self.use_precursor_mass = use_precursor_mass
        self.mz_max = mz_max

        self.peak_emb = PeakEmbedding(mz_bins, d_model, peak_embed_dim)
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len=max(512, max_smiles_len + 8), dropout=dropout)
        self.peak_pos = PositionalEncoding(d_model, max_len=512, dropout=dropout)
        # Light formula/mass conditioning: broadcast precursor m/z embedding onto peaks
        self.mass_proj = nn.Sequential(
            nn.Linear(1, peak_embed_dim),
            nn.GELU(),
            nn.Linear(peak_embed_dim, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        mz_bin_ids: torch.Tensor,
        intensity: torch.Tensor,
        mz_norm: torch.Tensor,
        peak_mask: torch.Tensor,
        precursor_mz: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.peak_emb(mz_bin_ids, intensity, mz_norm)
        if self.use_precursor_mass and precursor_mz is not None:
            # precursor_mz: [B] or [B, 1] → normalize by mz_max
            pmz = precursor_mz.view(-1).float() / max(self.mz_max, 1e-6)
            pmz = pmz.clamp(0.0, 2.0).unsqueeze(-1)  # [B, 1]
            mass_emb = self.mass_proj(pmz)  # [B, D]
            x = x + mass_emb.unsqueeze(1)
        x = self.peak_pos(x)
        memory = self.encoder(x, src_key_padding_mask=peak_mask)
        return memory

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        peak_mask: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.token_emb(tgt_ids) * math.sqrt(self.d_model)
        emb = self.pos_enc(emb)
        causal = nn.Transformer.generate_square_subsequent_mask(
            tgt_ids.size(1), device=tgt_ids.device, dtype=torch.bool
        )
        out = self.decoder(
            emb,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=peak_mask,
        )
        return self.out_proj(out)

    def forward(
        self,
        mz_bin_ids: torch.Tensor,
        intensity: torch.Tensor,
        mz_norm: torch.Tensor,
        peak_mask: torch.Tensor,
        smiles_ids: torch.Tensor,
        smiles_mask: Optional[torch.Tensor] = None,
        precursor_mz: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encode(
            mz_bin_ids, intensity, mz_norm, peak_mask, precursor_mz=precursor_mz
        )
        tgt_in = smiles_ids[:, :-1]
        tgt_pad = smiles_mask[:, :-1] if smiles_mask is not None else (tgt_in == self.pad_id)
        logits = self.decode(tgt_in, memory, peak_mask, tgt_key_padding_mask=tgt_pad)
        return logits

    def loss(
        self,
        logits: torch.Tensor,
        smiles_ids: torch.Tensor,
    ) -> torch.Tensor:
        target = smiles_ids[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=self.pad_id,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
