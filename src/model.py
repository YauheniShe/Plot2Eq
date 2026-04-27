import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dims))

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.w3.weight)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    B, L, H, D = xq.shape

    xq = xq.view(B, L, H, D // 2, 2)
    xk = xk.view(B, L, H, D // 2, 2)

    xq_r, xq_i = xq[..., 0], xq[..., 1]
    xk_r, xk_i = xk[..., 0], xk[..., 1]

    fc = freqs_cos.to(device=xq.device, dtype=xq.dtype)
    fs = freqs_sin.to(device=xq.device, dtype=xq.dtype)

    xq_out_r = xq_r * fc - xq_i * fs
    xq_out_i = xq_r * fs + xq_i * fc
    xk_out_r = xk_r * fc - xk_i * fs
    xk_out_i = xk_r * fs + xk_i * fc

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).view(B, L, H, D)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).view(B, L, H, D)

    return xq_out, xk_out


class Attention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1) -> None:
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        x,
        context=None,
        freqs_cos=None,
        freqs_sin=None,
        is_causal=False,
        kv_cache=None,
    ):
        B, L_q, D = x.shape

        is_cross_attn = context is not None

        q = self.q_proj(x).view(B, L_q, self.nhead, self.head_dim)

        if is_cross_attn and kv_cache is not None:
            k, v = kv_cache
            new_kv_cache = kv_cache
        else:
            context = x if context is None else context
            _, L_k, _ = context.shape

            k = self.k_proj(context).view(B, L_k, self.nhead, self.head_dim)
            v = self.v_proj(context).view(B, L_k, self.nhead, self.head_dim)

            if freqs_cos is not None and freqs_sin is not None:
                q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

            if kv_cache is not None and not is_cross_attn:
                past_k, past_v = kv_cache
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)

            new_kv_cache = (k, v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = self.dropout_p if self.training else 0.0

        actual_is_causal = is_causal if L_q > 1 else False

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=actual_is_causal, dropout_p=dropout_p
        )

        out = self.o_proj(attn_out.transpose(1, 2).contiguous().view(B, L_q, D))

        return out, new_kv_cache


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        # x shape: (B, SeqLen, d_model)
        return x + self.pe[:, : x.size(1), :]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, dim_feedforward, d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.self_attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1) -> None:
        super().__init__()
        self.self_attn = Attention(d_model, nhead, dropout=dropout)
        self.cross_attn = Attention(d_model, nhead, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, dim_feedforward, d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, freqs_cos, freqs_sin, kv_cache=None):
        sa_cache, ca_cache = kv_cache if kv_cache is not None else (None, None)

        sa_out, new_sa_cache = self.self_attn(
            self.norm1(x),
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            is_causal=True,
            kv_cache=sa_cache,
        )
        x = x + self.dropout(sa_out)

        ca_out, new_ca_cache = self.cross_attn(
            self.norm2(x), context=memory, kv_cache=ca_cache
        )
        x = x + self.dropout(ca_out)
        x = x + self.dropout(self.ff(self.norm3(x)))

        return x, (new_sa_cache, new_ca_cache)


class ConvNeXt1DBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x):

        residual = x
        x = self.dwconv(x)

        x = torch.permute(x, (0, 2, 1))
        x = self.net(x)

        x = self.gamma * x
        x = torch.permute(x, (0, 2, 1))

        return residual + x


class ConvNeXt1DEncoder(nn.Module):
    def __init__(self, in_channels=2, d_model=256):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, 64),
        )

        self.stage1 = nn.Sequential(ConvNeXt1DBlock(64), ConvNeXt1DBlock(64))

        self.down1 = nn.Sequential(
            nn.GroupNorm(1, 64), nn.Conv1d(64, 128, kernel_size=2, stride=2)
        )

        self.stage2 = nn.Sequential(ConvNeXt1DBlock(128), ConvNeXt1DBlock(128))

        self.down2 = nn.Sequential(
            nn.GroupNorm(1, 128), nn.Conv1d(128, 256, kernel_size=1, stride=1)
        )

        self.stage3 = nn.Sequential(ConvNeXt1DBlock(256), ConvNeXt1DBlock(256))

        self.down3 = nn.Sequential(
            nn.GroupNorm(1, 256), nn.Conv1d(256, d_model, kernel_size=1, stride=1)
        )

        self.stage4 = nn.Sequential(ConvNeXt1DBlock(d_model), ConvNeXt1DBlock(d_model))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        return x


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

        self.eval()
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

        self.eval()
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
