import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from plot2eq.models.components import (
    ConvNeXt1DEncoder,
    DecoderLayer,
    EncoderLayer,
    LearnablePositionalEncoding,
    RMSNorm,
    precompute_freqs_cis,
)


class Plot2EqModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_idx,
        d_model=512,
        nhead=8,
        num_enc_layers=4,
        num_dec_layers=4,
        max_seq_len=256,
        dropout=0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        self.cnn_encoder = ConvNeXt1DEncoder(in_channels=2, d_model=d_model)

        self.encoder_pe = LearnablePositionalEncoding(d_model, max_len=64)

        dim_feedforward = int(d_model * 8 / 3)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_enc_layers)
            ]
        )
        self.enc_norm = RMSNorm(d_model)

        self.target_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)

        nn.init.normal_(self.target_emb.weight, std=d_model**-0.5)

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_dec_layers)
            ]
        )
        self.dec_norm = RMSNorm(d_model)

        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)
        self.fc_out.weight = self.target_emb.weight

        head_dim = d_model // nhead
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, src, tgt):
        # src shape : (B, 2, 256)
        # target shape: (B, seq_len)

        features = self.cnn_encoder(src)
        memory = features.transpose(1, 2)
        memory = self.encoder_pe(memory)

        for layer in self.encoder_layers:
            memory = layer(memory)

        memory = self.enc_norm(memory)

        tgt_embedded = self.target_emb(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.emb_dropout(tgt_embedded)

        seq_len = tgt.size(1)

        head_dim = self.d_model // self.encoder_layers[0].self_attn.nhead  # type: ignore
        fc = self.freqs_cos[:seq_len].view(1, seq_len, 1, head_dim // 2)  # type: ignore
        fs = self.freqs_sin[:seq_len].view(1, seq_len, 1, head_dim // 2)  # type: ignore

        for layer in self.decoder_layers:
            tgt_embedded, _ = layer(
                x=tgt_embedded, memory=memory, freqs_cos=fc, freqs_sin=fs
            )

        tgt_embedded = self.dec_norm(tgt_embedded)
        logits = self.fc_out(tgt_embedded)
        return logits

    @torch.no_grad()
    def generate(self, src, sos_idx, eos_idx, max_len=128):

        B = src.size(0)
        device = src.device

        features = self.cnn_encoder(src)
        memory = features.transpose(1, 2)
        memory = self.encoder_pe(memory)
        for layer in self.encoder_layers:
            memory = layer(memory)
        memory = self.enc_norm(memory)

        tokens = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
        kv_caches = [None] * len(self.decoder_layers)

        head_dim = self.d_model // self.decoder_layers[0].self_attn.nhead  # type: ignore

        for pos in range(max_len):
            curr_token = tokens[:, -1:]

            tgt_embedded = self.target_emb(curr_token) * math.sqrt(self.d_model)
            fc = self.freqs_cos[pos : pos + 1].view(1, 1, 1, head_dim // 2)  # type: ignore
            fs = self.freqs_sin[pos : pos + 1].view(1, 1, 1, head_dim // 2)  # type: ignore

            new_caches = []
            for i, layer in enumerate(self.decoder_layers):
                tgt_embedded, new_layer_cache = layer(
                    x=tgt_embedded,
                    memory=memory,
                    freqs_cos=fc,
                    freqs_sin=fs,
                    kv_cache=kv_caches[i],
                )
                new_caches.append(new_layer_cache)

            kv_caches = new_caches

            tgt_embedded = self.dec_norm(tgt_embedded)
            logits = self.fc_out(tgt_embedded)  # (B, 1, vocab_size)

            next_token = logits.argmax(dim=-1)  # (B, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

            if (tokens == eos_idx).any(dim=1).all():
                break

        return tokens

    def beam_search(self, src, sos_idx, eos_idx, max_len=128, beam_size=5):

        B = src.size(0)
        device = src.device

        features = self.cnn_encoder(src)
        memory = features.transpose(1, 2)
        memory = self.encoder_pe(memory)
        for layer in self.encoder_layers:
            memory = layer(memory)
        memory = self.enc_norm(memory)

        memory = memory.repeat_interleave(beam_size, dim=0)
        tokens = torch.full(
            (B * beam_size, 1), sos_idx, dtype=torch.long, device=device
        )
        scores = torch.full((B, beam_size), -float("inf"), device=device)
        scores[:, 0] = 0.0
        scores = scores.view(-1)

        is_finished = torch.zeros(B * beam_size, dtype=torch.bool, device=device)

        kv_caches = [None] * len(self.decoder_layers)
        head_dim = self.d_model // self.decoder_layers[0].self_attn.nhead  # type: ignore

        for pos in range(max_len):
            curr_token = tokens[:, -1:]

            tgt_embedded = self.target_emb(curr_token) * math.sqrt(self.d_model)
            fc = self.freqs_cos[pos : pos + 1].view(1, 1, 1, head_dim // 2)  # type: ignore
            fs = self.freqs_sin[pos : pos + 1].view(1, 1, 1, head_dim // 2)  # type: ignore

            new_caches = []
            for i, layer in enumerate(self.decoder_layers):
                tgt_embedded, new_layer_cache = layer(
                    x=tgt_embedded,
                    memory=memory,
                    freqs_cos=fc,
                    freqs_sin=fs,
                    kv_cache=kv_caches[i],
                )
                new_caches.append(new_layer_cache)

            kv_caches = new_caches

            tgt_embedded = self.dec_norm(tgt_embedded)
            logits = self.fc_out(tgt_embedded).squeeze(
                1
            )  # (B * beam_width, vocab_size)
            vocab_size = logits.size(-1)

            log_probs = F.log_softmax(logits, dim=-1)

            log_probs[is_finished] = -float("inf")
            log_probs[is_finished, eos_idx] = 0.0

            next_scores = (
                scores.unsqueeze(-1) + log_probs
            )  # (B * beam_width, vocab_size)
            next_scores = next_scores.view(B, -1)  # (B, beam_width * vocab_size)

            topk_scores, topk_indices = torch.topk(next_scores, beam_size, dim=-1)

            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size

            batch_offsets = torch.arange(B, device=device).unsqueeze(1) * beam_size
            global_beam_indices = (batch_offsets + beam_indices).view(-1)

            tokens = torch.cat(
                [tokens[global_beam_indices], token_indices.view(-1, 1)], dim=1
            )
            scores = topk_scores.view(-1)
            is_finished = is_finished[global_beam_indices] | (
                token_indices.view(-1) == eos_idx
            )

            reordered_caches = []
            for sa_cache, ca_cache in kv_caches:
                sa_k, sa_v = sa_cache
                sa_k, sa_v = sa_k[global_beam_indices], sa_v[global_beam_indices]

                ca_k, ca_v = ca_cache
                ca_k, ca_v = ca_k[global_beam_indices], ca_v[global_beam_indices]

                reordered_caches.append(((sa_k, sa_v), (ca_k, ca_v)))

            kv_caches = reordered_caches

            if is_finished.all():
                break

        all_candidates = tokens.reshape(B, beam_size, -1)
        return all_candidates
