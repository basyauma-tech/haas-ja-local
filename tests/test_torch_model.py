from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from haas_ja_local.config import HaasConfig
from haas_ja_local.decode import greedy_tdt
from haas_ja_local.features import log_mel
from haas_ja_local.nemo_checkpoint import expected_keys
from haas_ja_local.torch_model import HaasTorch, RelPositionalEncoding, encoded_length, load_torch_model, rel_shift
from tests.conftest import make_state_dict


def _model_from_random(config: HaasConfig, seed: int = 0) -> HaasTorch:
    model = HaasTorch(config)
    model.load_state_dict({k: v.float() for k, v in make_state_dict(config, seed=seed).items()}, strict=True)
    return model.eval()


def test_state_dict_keys_match_checkpoint(tiny_config):
    model = HaasTorch(tiny_config)
    assert set(model.state_dict()) == expected_keys(tiny_config)


def test_state_dict_shapes_match_checkpoint(tiny_config, tiny_state_dict):
    model = HaasTorch(tiny_config)
    assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == {
        k: tuple(v.shape) for k, v in tiny_state_dict.items()
    }


def test_loads_tiny_nemo_strict(tiny_nemo, tiny_config, tiny_state_dict):
    model, tok = load_torch_model(tiny_nemo)
    assert model.config.d_model == tiny_config.d_model
    assert tok.vocab_size == tiny_config.vocab_size
    sd = model.state_dict()
    assert all(sd[k].dtype == torch.float32 for k in sd if not k.endswith("num_batches_tracked"))
    torch.testing.assert_close(sd["joint.enc.weight"], tiny_state_dict["joint.enc.weight"].float())
    assert model.window_np.dtype == np.float32 and model.window_np.shape == (tiny_config.win_length,)
    assert model.fb_np.dtype == np.float32
    assert model.fb_np.shape == (tiny_config.n_mels, tiny_config.n_fft // 2 + 1)
    assert not model.training


@pytest.mark.parametrize("n_frames", [16, 17, 31, 32, 33, 160])
def test_encoder_output_length(tiny_nemo, n_frames, tiny_config):
    model, _ = load_torch_model(tiny_nemo)
    mel = torch.randn(1, tiny_config.n_mels, -(-n_frames // 16) * 16)
    with torch.inference_mode():
        enc, lengths = model.encoder(mel, torch.tensor([n_frames]))
    assert enc.shape == (1, tiny_config.d_model, encoded_length(n_frames, tiny_config))
    assert int(lengths[0]) == enc.shape[2]
    assert torch.isfinite(enc).all()


def test_encoded_length_examples():
    assert [encoded_length(n, HaasConfig()) for n in (0, 1, 8, 9, 16, 17, 100)] == [0, 1, 1, 2, 2, 3, 13]


@pytest.mark.parametrize("n_frames", [33, 35, 38, 40, 41, 47, 48])
def test_encoder_ignores_padding_frames(tiny_nemo, tiny_config, n_frames):
    # NeMoのMaskedConvSequentialと同じく、サブサンプリングの各層の前で有効長より後ろを0にする。
    # そのため有効長の直後からmelを変えても出力は変わらない
    model, _ = load_torch_model(tiny_nemo)
    gen = torch.Generator().manual_seed(1)
    mel = torch.randn(1, tiny_config.n_mels, 64, generator=gen)
    other = mel.clone()
    other[..., n_frames:] = torch.randn(1, tiny_config.n_mels, 64 - n_frames, generator=gen)
    with torch.inference_mode():
        a, _ = model.encoder(mel, torch.tensor([n_frames]))
        b, _ = model.encoder(other, torch.tensor([n_frames]))
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("n_frames", [33, 34, 36, 40, 41, 48])
def test_encoder_padded_input_matches_unpadded(tiny_nemo, tiny_config, n_frames):
    # 0埋めしたmelでも、有効長で切り詰めたmelと同じ出力になる（中間段の有効長外もマスクされる）
    model, _ = load_torch_model(tiny_nemo)
    mel = torch.zeros(1, tiny_config.n_mels, 64)
    mel[..., :n_frames] = torch.randn(1, tiny_config.n_mels, n_frames, generator=torch.Generator().manual_seed(2))
    with torch.inference_mode():
        padded, _ = model.encoder(mel, torch.tensor([n_frames]))
        unpadded, _ = model.encoder(mel[..., :n_frames], torch.tensor([n_frames]))
    torch.testing.assert_close(padded, unpadded, rtol=1e-5, atol=1e-5)


def test_conv_subsampling_flattens_channel_then_freq(tiny_config):
    # tinyのn_mels=8ではサブサンプリング後の周波数が1になるので、32にして平坦化順（チャンネル→周波数）を確かめる
    c = replace(tiny_config, n_mels=32)
    model = _model_from_random(c, seed=11)
    sub = model.encoder.pre_encode
    x = torch.randn(1, 20, c.n_mels, generator=torch.Generator().manual_seed(12))
    conv = sub.conv
    h = F.relu(F.conv2d(x[:, None], conv[0].weight, conv[0].bias, stride=2, padding=1))
    for dw, pw in ((conv[2], conv[3]), (conv[5], conv[6])):
        h = F.conv2d(h, dw.weight, dw.bias, stride=2, padding=1, groups=c.subsampling_channels)
        h = F.relu(F.conv2d(h, pw.weight, pw.bias))
    assert h.shape[1:] == (c.subsampling_channels, 3, 4)
    w = sub.out.weight.view(c.d_model, c.subsampling_channels, 4)  # 入力列はch*4+freq
    expected = torch.einsum("bctf,dcf->btd", h, w) + sub.out.bias
    with torch.inference_mode():
        torch.testing.assert_close(sub(x, torch.tensor([20])), expected, rtol=1e-5, atol=1e-5)


def test_rel_positional_encoding_values():
    d, t = 6, 4
    x = torch.ones(1, t, d)
    scaled, pe = RelPositionalEncoding(d)(x)
    torch.testing.assert_close(scaled, x * d**0.5)
    assert pe.shape == (1, 2 * t - 1, d)
    for row, rel in enumerate(range(t - 1, -t, -1)):  # 先頭が相対位置T-1、末尾が-(T-1)
        for i in range(0, d, 2):
            angle = torch.tensor(rel / 10000 ** (i / d))
            torch.testing.assert_close(pe[0, row, i], torch.sin(angle))
            torch.testing.assert_close(pe[0, row, i + 1], torch.cos(angle))


def test_rel_shift_aligns_relative_positions():
    # bd[i, j]は相対位置i-j（pos_embの行T-1-(i-j)）とのスコアになる
    t = 5
    scores = torch.randn(2, 3, t, 2 * t - 1, generator=torch.Generator().manual_seed(2))
    shifted = rel_shift(scores)[..., :t]
    for i in range(t):
        for j in range(t):
            torch.testing.assert_close(shifted[..., i, j], scores[..., i, t - 1 - (i - j)])


def test_rel_position_attention_matches_explicit_indexing(tiny_config):
    # rel_shiftを使わず、相対位置i-jに対応するpos_embの行を直接引いて計算する
    c = tiny_config
    model = _model_from_random(c, seed=13)
    attn = model.encoder.layers[0].self_attn
    t, h, dk = 6, c.n_heads, c.d_model // c.n_heads
    x = torch.randn(1, t, c.d_model, generator=torch.Generator().manual_seed(14))
    _, pos_emb = model.encoder.pos_enc(x)
    q = F.linear(x, attn.linear_q.weight).view(t, h, dk)
    k = F.linear(x, attn.linear_k.weight).view(t, h, dk)
    v = F.linear(x, attn.linear_v.weight).view(t, h, dk)
    p = F.linear(pos_emb[0], attn.linear_pos.weight).view(2 * t - 1, h, dk)
    scores = torch.empty(h, t, t)
    for i in range(t):
        for j in range(t):
            content = ((q[i] + attn.pos_bias_u) * k[j]).sum(-1)
            position = ((q[i] + attn.pos_bias_v) * p[t - 1 - (i - j)]).sum(-1)
            scores[:, i, j] = (content + position) / dk**0.5
    out = torch.einsum("hij,jhd->ihd", scores.softmax(-1), v).reshape(1, t, c.d_model)
    expected = F.linear(out, attn.linear_out.weight)
    with torch.inference_mode():
        torch.testing.assert_close(attn(x, pos_emb), expected, rtol=1e-5, atol=1e-5)


def _naive_kda_sequence(layer, x: torch.Tensor, n_heads: int, head_dim: int, conv_size: int):
    """fla の fused_recurrent_kda をそのまま書いた素朴実装。状態は [B,H,K,V]（k行・v列）。"""
    b, u, _ = x.shape
    d = n_heads * head_dim

    def short_conv(w_proj: torch.Tensor, w_conv: torch.Tensor) -> torch.Tensor:
        h = F.linear(x, w_proj).transpose(1, 2)  # [B,D,U]
        h = F.conv1d(h, w_conv, padding=conv_size - 1, groups=d)[..., :u]
        return F.silu(h).transpose(1, 2).reshape(b, u, n_heads, head_dim)

    q = short_conv(layer.q_proj.weight, layer.q_conv1d.weight)
    k = short_conv(layer.k_proj.weight, layer.k_conv1d.weight)
    v = short_conv(layer.v_proj.weight, layer.v_conv1d.weight)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * head_dim**-0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    g = F.linear(F.linear(x, layer.f_proj[0].weight), layer.f_proj[1].weight).reshape(b, u, n_heads, head_dim)
    g = -layer.A_log.exp()[:, None] * F.softplus(g + layer.dt_bias.reshape(n_heads, head_dim))
    beta = torch.sigmoid(F.linear(x, layer.b_proj.weight))  # [B,U,H]

    s = torch.zeros(b, n_heads, head_dim, head_dim)
    outs = []
    for t in range(u):
        q_t, k_t, v_t, g_t, b_t = q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t]
        s = s * g_t.exp()[..., None]
        v_new = (v_t - (k_t[..., None] * s).sum(-2)) * b_t[..., None]
        s = s + k_t[..., None] * v_new[..., None, :]
        outs.append((q_t[..., None] * s).sum(-2))
    o = torch.stack(outs, dim=1)  # [B,U,H,V]

    gate = F.linear(F.linear(x, layer.g_proj[0].weight), layer.g_proj[1].weight, layer.g_proj[1].bias)
    gate = gate.reshape(b, u, n_heads, head_dim)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-5) * layer.o_norm.weight
    o = o * torch.sigmoid(gate)
    return F.linear(o.reshape(b, u, d), layer.o_proj.weight), s


def test_kda_step_sequence_matches_batched_naive(tiny_config):
    c = tiny_config
    model = _model_from_random(c, seed=3)
    layer = model.decoder.prediction.dec_rnn.layers[0]
    b, u, d = 2, 5, c.kda_heads * c.kda_head_dim
    x = torch.randn(b, u, c.pred_hidden, generator=torch.Generator().manual_seed(4))

    with torch.inference_mode():
        expected, expected_state = _naive_kda_sequence(layer, x, c.kda_heads, c.kda_head_dim, c.kda_conv_size)
        rec = torch.zeros(b, c.kda_heads, c.kda_head_dim, c.kda_head_dim)
        conv = tuple(torch.zeros(b, d, c.kda_conv_size) for _ in range(3))
        ys = []
        for t in range(u):
            y, rec, conv = layer(x[:, t : t + 1], rec, conv)
            assert y.shape == (b, 1, c.pred_hidden)
            ys.append(y)
    torch.testing.assert_close(torch.cat(ys, dim=1), expected, rtol=1e-5, atol=1e-5)
    # モデルの状態は [B,H,V,K]、素朴実装は [B,H,K,V]
    torch.testing.assert_close(rec, expected_state.transpose(-1, -2), rtol=1e-5, atol=1e-5)


def test_joint_formula(tiny_config):
    c = tiny_config
    model = _model_from_random(c, seed=5)
    gen = torch.Generator().manual_seed(6)
    f = torch.randn(3, c.d_model, generator=gen)
    g = torch.randn(3, c.pred_hidden, generator=gen)
    j = model.joint
    hidden = F.relu(F.linear(f, j.enc.weight, j.enc.bias) + F.linear(g, j.pred.weight, j.pred.bias))
    expected = F.linear(hidden, j.joint_net[2].weight, j.joint_net[2].bias)
    with torch.inference_mode():
        logits = j(f, g)
    assert logits.shape == (3, c.vocab_size + 1 + len(c.durations))
    torch.testing.assert_close(logits, expected)


def test_prediction_network_matches_layerwise_reference(tiny_config):
    # 予測ネットワーク: embed → 各層 x + layer(norms[i](x)) → final_norm。状態の詰め方も確かめる
    c = tiny_config
    model = _model_from_random(c, seed=7)
    pred = model.decoder.prediction
    rnn = pred.dec_rnn
    d = c.kda_heads * c.kda_head_dim
    tokens = torch.tensor([3, c.blank_id])
    with torch.inference_mode():
        x = pred.embed(tokens)[:, None]
        recs, convs = [], []
        for i in range(c.pred_layers):
            rec = torch.zeros(2, c.kda_heads, c.kda_head_dim, c.kda_head_dim)
            conv = tuple(torch.zeros(2, d, c.kda_conv_size) for _ in range(3))
            y, rec, conv = rnn.layers[i](
                F.layer_norm(x, (c.pred_hidden,), rnn.norms[i].weight, rnn.norms[i].bias), rec, conv
            )
            x = x + y
            recs.append(rec)
            convs.append(conv)
        expected = F.layer_norm(x[:, 0], (c.pred_hidden,), rnn.final_norm.weight, rnn.final_norm.bias)

        width = c.kda_state_width
        zeros = torch.zeros(c.pred_layers, 2, width)
        g, rec_packed, conv_packed = pred(tokens, zeros, zeros.clone())
    torch.testing.assert_close(g, expected)
    assert rec_packed.shape == conv_packed.shape == (c.pred_layers, 2, width)
    n_rec = c.kda_heads * c.kda_head_dim**2
    n_conv = 3 * d * c.kda_conv_size
    for i in range(c.pred_layers):
        torch.testing.assert_close(rec_packed[i, :, :n_rec], recs[i].reshape(2, -1))
        conv_flat = torch.cat([t.reshape(2, -1) for t in convs[i]], dim=-1)
        torch.testing.assert_close(conv_packed[i, :, :n_conv], conv_flat)
    assert not rec_packed[:, :, n_rec:].any()
    assert not conv_packed[:, :, n_conv:].any()


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # 畳み込み状態の方が広い（tiny）
        {"kda_heads": 1, "kda_head_dim": 16},  # 再帰状態の方が広い（実モデルと同じ向き）
    ],
)
def test_step_state_packing_roundtrip(tiny_config, overrides):
    c = replace(tiny_config, **overrides)
    model = _model_from_random(c, seed=8)
    enc = torch.randn(4, c.d_model, generator=torch.Generator().manual_seed(9))
    state = model.init_state()
    assert [tuple(s.shape) for s in state] == [(c.pred_layers, 1, c.kda_state_width)] * 2
    tokens = [1, 5, 2]
    for t, token in enumerate(tokens):
        logits, state = model.step(enc, t, token, state)
    # 詰めた状態を経由した結果が、状態を持ち回さずに1回で流した結果と一致する
    with torch.inference_mode():
        pred = model.decoder.prediction
        rec, conv = (s.clone() for s in model.init_state())
        for token in tokens:
            g, rec, conv = pred(torch.tensor([token]), rec, conv)
        expected = model.joint(enc[len(tokens) - 1][None], g)[0].numpy()
    np.testing.assert_allclose(logits, expected, rtol=1e-6, atol=1e-6)
    n_rec = c.kda_heads * c.kda_head_dim**2
    n_conv = 3 * c.kda_heads * c.kda_head_dim * c.kda_conv_size
    assert state[0][..., :n_rec].any() and not state[0][..., n_rec:].any()
    assert state[1][..., :n_conv].any() and not state[1][..., n_conv:].any()


def test_step_is_deterministic_and_token_dependent(tiny_nemo, tiny_config):
    model, _ = load_torch_model(tiny_nemo)
    enc = torch.randn(3, tiny_config.d_model, generator=torch.Generator().manual_seed(10))
    state = model.init_state()
    a1, s1 = model.step(enc, 1, 4, state)
    a2, s2 = model.step(enc, 1, 4, state)
    b, _ = model.step(enc, 1, 5, state)
    assert a1.dtype == np.float32 and a1.shape == (tiny_config.vocab_size + 1 + len(tiny_config.durations),)
    np.testing.assert_array_equal(a1, a2)
    for x, y in zip(s1, s2, strict=True):
        assert torch.equal(x, y)
    assert not np.allclose(a1, b)
    # 入力の状態は書き換えない
    assert not state[0].any() and not state[1].any()


def test_encode_returns_truncated_frames(tiny_nemo, tiny_config):
    model, _ = load_torch_model(tiny_nemo)
    samples = np.random.default_rng(1).standard_normal(8000).astype(np.float32)
    mel, n = log_mel(samples, model.window_np, model.fb_np, model.config)
    encoded = model.encode(mel, n)
    assert isinstance(encoded, torch.Tensor) and encoded.dtype == torch.float32
    assert encoded.shape == (encoded_length(n, tiny_config), tiny_config.d_model)
    assert model.num_frames(encoded) == encoded.shape[0]
    empty = model.encode(np.zeros((tiny_config.n_mels, 0), dtype=np.float32), 0)
    assert model.num_frames(empty) == 0


def test_backend_protocol_end_to_end(tiny_nemo):
    model, _tok = load_torch_model(tiny_nemo)
    mel, n = log_mel(
        np.random.default_rng(0).standard_normal(16000).astype(np.float32),
        model.window_np,
        model.fb_np,
        model.config,
    )
    tokens = greedy_tdt(model, model.encode(mel, n))
    assert all(0 <= t.id < model.config.vocab_size for t in tokens)
