"""Haas-ja-v1.0の純PyTorchリファレンス実装（推論専用、単一発話）。

MLX・ONNXの各バックエンドはこの実装の出力を正として検証する。モジュール名は元チェックポイントの
state dictのキーと完全に一致させてあり、`load_state_dict(strict=True)`で読める。
torchはconvert extraでのみ入る。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from haas_ja_local.architecture import KDA_L2_EPS, KDA_RMS_EPS, LAYER_NORM_EPS, subsampling_stages
from haas_ja_local.config import HaasConfig
from haas_ja_local.decode import Backend
from haas_ja_local.nemo_checkpoint import NemoCheckpoint, read_nemo
from haas_ja_local.tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# エンコーダ（FastConformer）


class ConvSubsampling(nn.Module):
    """dw_stridingのサブサンプリング。キーはpre_encode.conv.{0,2,3,5,6}とpre_encode.out。"""

    def __init__(self, config: HaasConfig) -> None:
        super().__init__()
        ch = config.subsampling_channels
        layers: list[nn.Module] = [nn.Conv2d(1, ch, 3, stride=2, padding=1), nn.ReLU()]
        freq = (config.n_mels + 1) // 2
        for _ in range(subsampling_stages(config) - 1):
            layers += [
                nn.Conv2d(ch, ch, 3, stride=2, padding=1, groups=ch),
                nn.Conv2d(ch, ch, 1),
                nn.ReLU(),
            ]
            freq = (freq + 1) // 2
        self.conv = nn.Sequential(*layers)
        self.out = nn.Linear(ch * freq, config.d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x [B,T,F]、lengths [B] → [B,ceil(T/8),d_model]。

        NeMoのMaskedConvSequentialと同じく、各層の前で時間軸の有効長より後ろを0にする。
        有効長はstride 2の畳み込みごとにceil(L/2)へ更新する。これをしないと、有効長が8の倍数でないとき
        最後の有効フレームが0埋め部分（バイアスとReLUで0でなくなった値）を読み、自己注意で全フレームへ広がる。
        """
        x = x.unsqueeze(1)  # [B,1,T,F]
        lengths = lengths.to(torch.int64)
        for layer in self.conv:
            x = x * _time_mask(x, lengths)
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and layer.stride[0] != 1:
                lengths = (lengths + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
        x = x * _time_mask(x, lengths)  # [B,C,T',F']
        b, _, t, _ = x.shape
        x = x.transpose(1, 2).reshape(b, t, -1)  # チャンネルが先、周波数が後
        return self.out(x)


def _time_mask(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """x [B,C,T,F]に掛ける時間軸のマスク[B,1,T,1]。"""
    t = torch.arange(x.shape[2], device=x.device)
    return (t[None, :] < lengths[:, None]).to(x.dtype)[:, None, :, None]


class RelPositionalEncoding(nn.Module):
    """xscalingと相対位置埋め込み。パラメータなし。"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = x.shape[1]
        pos = torch.arange(t - 1, -t, -1, dtype=torch.float32, device=x.device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=x.device) * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(2 * t - 1, self.d_model, dtype=torch.float32, device=x.device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return x * self.xscale, pe.unsqueeze(0).to(x.dtype)


def rel_shift(x: torch.Tensor) -> torch.Tensor:
    """x [B,H,T,2T-1]の各行iを、相対位置i-jが列jに来るようずらす。"""
    b, h, qlen, pos_len = x.shape
    x = F.pad(x, (1, 0))
    x = x.view(b, h, -1, qlen)
    return x[:, :, 1:].reshape(b, h, qlen, pos_len)


class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.d_k))

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.linear_q(x).view(b, t, self.h, self.d_k).transpose(1, 2)  # [B,H,T,dk]
        k = self.linear_k(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(pos_emb.shape[0], -1, self.h, self.d_k).transpose(1, 2)  # [1,H,2T-1,dk]
        q_u = q + self.pos_bias_u[None, :, None, :]
        q_v = q + self.pos_bias_v[None, :, None, :]
        ac = q_u @ k.transpose(-2, -1)  # [B,H,T,T]
        bd = rel_shift(q_v @ p.transpose(-2, -1))[..., :t]  # [B,H,T,T]
        scores = (ac + bd) / math.sqrt(self.d_k)
        attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        out = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return self.linear_out(out)


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.silu(self.linear1(x)))


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int) -> None:
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2, groups=d_model, bias=False
        )
        self.batch_norm = nn.BatchNorm1d(d_model, eps=1e-5)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B,T,d] → [B,T,d]。"""
        x = x.transpose(1, 2)
        x = F.glu(self.pointwise_conv1(x), dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class ConformerLayer(nn.Module):
    def __init__(self, config: HaasConfig) -> None:
        super().__init__()
        d = config.d_model
        self.norm_feed_forward1 = nn.LayerNorm(d, eps=LAYER_NORM_EPS)
        self.feed_forward1 = ConformerFeedForward(d, d * config.ff_expansion)
        self.norm_self_att = nn.LayerNorm(d, eps=LAYER_NORM_EPS)
        self.self_attn = RelPositionMultiHeadAttention(d, config.n_heads)
        self.norm_conv = nn.LayerNorm(d, eps=LAYER_NORM_EPS)
        self.conv = ConformerConvolution(d, config.conv_kernel_size)
        self.norm_feed_forward2 = nn.LayerNorm(d, eps=LAYER_NORM_EPS)
        self.feed_forward2 = ConformerFeedForward(d, d * config.ff_expansion)
        self.norm_out = nn.LayerNorm(d, eps=LAYER_NORM_EPS)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        r = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        r = r + self.self_attn(self.norm_self_att(r), pos_emb)
        r = r + self.conv(self.norm_conv(r))
        r = r + 0.5 * self.feed_forward2(self.norm_feed_forward2(r))
        return self.norm_out(r)


class ConformerEncoder(nn.Module):
    def __init__(self, config: HaasConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_encode = ConvSubsampling(config)
        self.pos_enc = RelPositionalEncoding(config.d_model)
        self.layers = nn.ModuleList(ConformerLayer(config) for _ in range(config.n_layers))

    def forward(self, mel: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """mel [B,n_mels,T], length [B] → (enc [B,d_model,T'], lengths [B])。

        サブサンプリングの中はNeMoと同じく有効長でマスクする。その後の自己注意はマスクを使わず、
        最長の有効長で切り詰める。単一発話（B=1）での利用を前提とし、長さの異なる発話をまとめて渡すと
        短い発話の出力はNeMoと一致しない。
        """
        lengths = length.to(torch.int64)
        x = self.pre_encode(mel.transpose(1, 2), lengths)
        for _ in range(subsampling_stages(self.config)):
            lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        x = x[:, : int(lengths.max())]
        x, pos_emb = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, pos_emb)
        return x.transpose(1, 2), lengths


# ---------------------------------------------------------------------------
# 予測ネットワーク（Kimi Delta Attention）


class _RMSNormWeight(nn.Module):
    """o_normの重みだけを持つ。正規化とゲートはKimiDeltaAttention側で行う。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class KimiDeltaAttention(nn.Module):
    """KDA層の1ステップ再帰形。状態S [B,H,V,K]（v行・k列）と短い畳み込みのキャッシュ（直近W入力）を持ち回す。"""

    def __init__(self, config: HaasConfig) -> None:
        super().__init__()
        p, h, k, w = config.pred_hidden, config.kda_heads, config.kda_head_dim, config.kda_conv_size
        d = h * k
        self.n_heads, self.head_dim, self.conv_size = h, k, w
        self.A_log = nn.Parameter(torch.zeros(h))
        self.dt_bias = nn.Parameter(torch.zeros(d))
        self.q_proj = nn.Linear(p, d, bias=False)
        self.k_proj = nn.Linear(p, d, bias=False)
        self.v_proj = nn.Linear(p, d, bias=False)
        self.q_conv1d = nn.Conv1d(d, d, w, groups=d, bias=False)
        self.k_conv1d = nn.Conv1d(d, d, w, groups=d, bias=False)
        self.v_conv1d = nn.Conv1d(d, d, w, groups=d, bias=False)
        self.f_proj = nn.Sequential(nn.Linear(p, k, bias=False), nn.Linear(k, d, bias=False))
        self.b_proj = nn.Linear(p, h, bias=False)
        self.g_proj = nn.Sequential(nn.Linear(p, k, bias=False), nn.Linear(k, d, bias=True))
        self.o_norm = _RMSNormWeight(k)
        self.o_proj = nn.Linear(d, p, bias=False)

    @staticmethod
    def _short_conv(h: torch.Tensor, conv: nn.Conv1d, cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """因果depthwise畳み込みの1ステップ。h [B,D]、cache [B,D,W]（古い順）→ (silu(y) [B,D], 新しいcache)。"""
        seq = torch.cat([cache[..., 1:], h.unsqueeze(-1)], dim=-1)
        y = (seq * conv.weight[:, 0, :].float()).sum(-1)
        return F.silu(y), seq

    def forward(
        self,
        x: torch.Tensor,
        rec: torch.Tensor,
        conv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """x [B,1,P]、rec [B,H,V,K]、conv（q,k,v各[B,D,W]）→ (y [B,1,P], rec', conv')。"""
        b = x.shape[0]
        h, kd = self.n_heads, self.head_dim
        x = x[:, 0]
        q, q_cache = self._short_conv(self.q_proj(x).float(), self.q_conv1d, conv[0].float())
        k, k_cache = self._short_conv(self.k_proj(x).float(), self.k_conv1d, conv[1].float())
        v, v_cache = self._short_conv(self.v_proj(x).float(), self.v_conv1d, conv[2].float())
        q = q.view(b, h, kd)
        k = k.view(b, h, kd)
        v = v.view(b, h, kd)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + KDA_L2_EPS) * kd**-0.5
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + KDA_L2_EPS)

        g = self.f_proj(x).float().view(b, h, kd)
        gk = -torch.exp(self.A_log.float())[None, :, None] * F.softplus(g + self.dt_bias.float().view(h, kd))
        beta = torch.sigmoid(self.b_proj(x).float())  # [B,H]

        s = rec.float() * torch.exp(gk)[:, :, None, :]
        v_new = (v - (s * k[:, :, None, :]).sum(-1)) * beta[..., None]
        s = s + v_new[..., None] * k[:, :, None, :]
        o = (s * q[:, :, None, :]).sum(-1)  # [B,H,V]

        gate = self.g_proj(x).float().view(b, h, kd)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + KDA_RMS_EPS) * self.o_norm.weight.float()
        o = o * torch.sigmoid(gate)
        y = self.o_proj(o.reshape(b, h * kd).to(x.dtype))
        return y.unsqueeze(1), s, (q_cache, k_cache, v_cache)


class _KDAStack(nn.Module):
    """dec_rnn: layers、norms（pre-norm）、final_norm。"""

    def __init__(self, config: HaasConfig) -> None:
        super().__init__()
        p = config.pred_hidden
        self.layers = nn.ModuleList(KimiDeltaAttention(config) for _ in range(config.pred_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(p, eps=LAYER_NORM_EPS) for _ in range(config.pred_layers))
        self.final_norm = nn.LayerNorm(p, eps=LAYER_NORM_EPS)


class KDAPredictionNetwork(nn.Module):
    """embed → 各層でx + layer(norms[i](x)) → final_norm。状態は(rec, conv)各[L,B,kda_state_width]に詰める。

    rec行はS [H,V,K]をその順に平坦化、conv行はq・k・vのキャッシュ[D,W]を順に平坦化して連結し、残りは0。
    """

    def __init__(self, config: HaasConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size + 1, config.pred_hidden)
        self.dec_rnn = _KDAStack(config)

    @property
    def _sizes(self) -> tuple[int, int]:
        c = self.config
        d = c.kda_heads * c.kda_head_dim
        return d * c.kda_head_dim, 3 * d * c.kda_conv_size

    def _unpack(
        self, rec: torch.Tensor, conv: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        c = self.config
        b = rec.shape[0]
        n_rec, n_conv = self._sizes
        s = rec[:, :n_rec].reshape(b, c.kda_heads, c.kda_head_dim, c.kda_head_dim)
        q, k, v = conv[:, :n_conv].reshape(b, 3, c.kda_heads * c.kda_head_dim, c.kda_conv_size).unbind(1)
        return s, (q, k, v)

    def _pack(
        self, s: torch.Tensor, caches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = s.shape[0]
        width = self.config.kda_state_width
        n_rec, n_conv = self._sizes
        rec = F.pad(s.reshape(b, n_rec), (0, width - n_rec))
        conv = F.pad(torch.cat([t.reshape(b, -1) for t in caches], dim=-1), (0, width - n_conv))
        return rec, conv

    def forward(
        self, token: torch.Tensor, rec: torch.Tensor, conv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """token [B]、rec・conv [L,B,W] → (g [B,P], rec', conv')。"""
        x = self.embed(token).unsqueeze(1)  # [B,1,P]
        new_rec, new_conv = [], []
        for i, (layer, norm) in enumerate(zip(self.dec_rnn.layers, self.dec_rnn.norms, strict=True)):
            s, caches = self._unpack(rec[i], conv[i])
            y, s, caches = layer(norm(x), s, caches)
            x = x + y
            r, c = self._pack(s, caches)
            new_rec.append(r)
            new_conv.append(c)
        return self.dec_rnn.final_norm(x[:, 0]), torch.stack(new_rec), torch.stack(new_conv)


class _Decoder(nn.Module):
    def __init__(self, config: HaasConfig) -> None:
        super().__init__()
        self.prediction = KDAPredictionNetwork(config)


# ---------------------------------------------------------------------------
# Jointと全体


class Joint(nn.Module):
    def __init__(self, config: HaasConfig) -> None:
        super().__init__()
        n_outputs = config.vocab_size + 1 + len(config.durations)
        self.pred = nn.Linear(config.pred_hidden, config.joint_hidden)
        self.enc = nn.Linear(config.d_model, config.joint_hidden)
        self.joint_net = nn.Sequential(nn.ReLU(), nn.Identity(), nn.Linear(config.joint_hidden, n_outputs))

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """f [B,d_model]、g [B,P] → logits [B,vocab_size+1+len(durations)]。"""
        return self.joint_net(self.enc(f) + self.pred(g))


class _Featurizer(nn.Module):
    def __init__(self, config: HaasConfig) -> None:
        super().__init__()
        self.register_buffer("window", torch.zeros(config.win_length))
        self.register_buffer("fb", torch.zeros(1, config.n_mels, config.n_fft // 2 + 1))


class _Preprocessor(nn.Module):
    def __init__(self, config: HaasConfig) -> None:
        super().__init__()
        self.featurizer = _Featurizer(config)


class HaasTorch(nn.Module, Backend):
    """Backendの実装。encodeはnumpyのmelを受けてTensor [T',d_model]を返し、stepはnumpyのlogitsを返す。"""

    def __init__(self, config: HaasConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.preprocessor = _Preprocessor(config)
        self.encoder = ConformerEncoder(config)
        self.decoder = _Decoder(config)
        self.joint = Joint(config)

    @property
    def window_np(self) -> np.ndarray:
        return self.preprocessor.featurizer.window.detach().float().cpu().numpy().copy()

    @property
    def fb_np(self) -> np.ndarray:
        return self.preprocessor.featurizer.fb.detach().float().cpu().numpy()[0].copy()

    @property
    def _dtype(self) -> torch.dtype:
        return self.joint.enc.weight.dtype

    def encode(self, mel: np.ndarray, n_frames: int) -> torch.Tensor:
        if n_frames <= 0:
            return torch.zeros(0, self.config.d_model, dtype=torch.float32)
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32)).unsqueeze(0).to(self._dtype)
            enc, _ = self.encoder(x, torch.tensor([n_frames]))
            return enc[0].transpose(0, 1).float().contiguous()

    def num_frames(self, encoded: torch.Tensor) -> int:
        return int(encoded.shape[0])

    def init_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (self.config.pred_layers, 1, self.config.kda_state_width)
        return torch.zeros(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.float32)

    def step(
        self, encoded: torch.Tensor, t: int, token: int, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor]]:
        with torch.inference_mode():
            rec, conv = state
            g, rec, conv = self.decoder.prediction(torch.tensor([token]), rec, conv)
            logits = self.joint(encoded[t].unsqueeze(0).to(self._dtype), g)[0]
            return logits.float().numpy(), (rec, conv)


def build_torch_model(ckpt: NemoCheckpoint, *, dtype: torch.dtype = torch.float32) -> HaasTorch:
    """読み込み済みのチェックポイントからHaasTorchを作る。bf16の重みはdtypeに変換する。

    キーや形状が合わなければload_state_dict(strict=True)のRuntimeError。
    """
    model = HaasTorch(ckpt.config).to(dtype)
    state_dict = {k: v.to(dtype) if v.is_floating_point() else v for k, v in ckpt.state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


def load_torch_model(nemo: Path, *, dtype: torch.dtype = torch.float32) -> tuple[HaasTorch, Tokenizer]:
    """.nemoからHaasTorchとトークナイザを作る。bf16の重みはdtype（既定float32）に変換して読む。"""
    ckpt = read_nemo(Path(nemo))
    return build_torch_model(ckpt, dtype=dtype), Tokenizer(ckpt.tokenizer_model)
