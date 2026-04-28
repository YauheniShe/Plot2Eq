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
