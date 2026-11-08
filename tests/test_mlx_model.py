import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from haas_ja_local.architecture import encoded_length  # noqa: E402
from haas_ja_local.convert.mlx import convert_nemo_to_mlx, expected_mlx_shapes  # noqa: E402
from haas_ja_local.decode import greedy_tdt  # noqa: E402
from haas_ja_local.mlx_model import HaasMLX  # noqa: E402
from haas_ja_local.pipeline import load_model  # noqa: E402
from haas_ja_local.torch_model import load_torch_model  # noqa: E402

# float32同士の比較（実測は6e-8程度）。CPU・GPUで和の順序が違っても通る幅
RTOL = ATOL = 1e-5


@pytest.fixture(scope="module")
def tiny_mlx_dir(tiny_nemo, tmp_path_factory):
    out = tmp_path_factory.mktemp("mlx")
    convert_nemo_to_mlx(tiny_nemo, out)
    return out


@pytest.fixture(scope="module")
def parity_mlx_dir(parity_nemo, tmp_path_factory):
    out = tmp_path_factory.mktemp("parity-mlx")
    convert_nemo_to_mlx(parity_nemo, out)
    return out


@pytest.fixture(scope="module")
def models(parity_nemo, parity_mlx_dir):
    """(torch float32, MLX float32)。"""
    torch_model, _ = load_torch_model(parity_nemo)
    return torch_model, HaasMLX.load(parity_mlx_dir, compute_dtype=mx.float32)


def _mel(config, n_frames: int, seed: int = 0) -> np.ndarray:
    padded = -(-n_frames // config.pad_to) * config.pad_to
    return np.random.default_rng(seed).standard_normal((config.n_mels, padded)).astype(np.float32)


def test_parameter_tree_matches_converted_keys_and_shapes(tiny_mlx_dir, tiny_config):
    # fail-closed: MLXのモジュール木（BatchNormの統計、窓とメルフィルタバンクも含む）が変換後のキーと形状に一致する
    model = HaasMLX.load(tiny_mlx_dir)
    params = dict(tree_flatten(model.module.parameters()))
    assert {k: tuple(v.shape) for k, v in params.items()} == expected_mlx_shapes(tiny_config)


def test_load_rejects_missing_weight(tiny_mlx_dir, tmp_path):
    import shutil

    from safetensors.torch import load_file, save_file

    for name in ("config.json", "tokenizer.model"):
        shutil.copy(tiny_mlx_dir / name, tmp_path / name)
    tensors = load_file(tiny_mlx_dir / "model.safetensors")
    del tensors["encoder.layers.0.conv.batch_norm.running_var"]
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="running_var"):
        HaasMLX.load(tmp_path)


def test_load_model_dispatches_directory_to_mlx(tiny_mlx_dir, tiny_config):
    backend, tokenizer = load_model(tiny_mlx_dir)
    assert isinstance(backend, HaasMLX)
    assert backend.config.d_model == tiny_config.d_model
    assert tokenizer.vocab_size == tiny_config.vocab_size
    assert backend.window_np.dtype == np.float32 and backend.window_np.shape == (tiny_config.win_length,)
    assert backend.fb_np.shape == (tiny_config.n_mels, tiny_config.n_fft // 2 + 1)


def test_window_and_fb_match_torch(models):
    torch_model, mlx_model = models
    np.testing.assert_array_equal(mlx_model.window_np, torch_model.window_np)
    np.testing.assert_array_equal(mlx_model.fb_np, torch_model.fb_np)


@pytest.mark.parametrize("n_frames", [100, 101, 97, 33, 8, 1])
def test_mlx_encoder_matches_torch_on_tiny_model(models, parity_config, n_frames):
    torch_model, mlx_model = models
    mel = _mel(parity_config, n_frames)
    expected = torch_model.encode(mel, n_frames).numpy()
    got = mlx_model.encode(mel, n_frames)
    assert isinstance(got, mx.array)
    assert got.shape == (encoded_length(n_frames, parity_config), parity_config.d_model)
    assert mlx_model.num_frames(got) == got.shape[0]
    np.testing.assert_allclose(np.array(got), expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n_frames", [33, 35, 38, 41, 47])
def test_mlx_encoder_ignores_padding_frames(models, parity_config, n_frames):
    # サブサンプリングの各層の前で有効長より後ろを0にする（torch_model.ConvSubsamplingと同じ）
    _, mlx_model = models
    mel = _mel(parity_config, 64, seed=1)
    other = mel.copy()
    other[:, n_frames:] = np.random.default_rng(2).standard_normal((parity_config.n_mels, 64 - n_frames))
    np.testing.assert_array_equal(
        np.array(mlx_model.encode(mel, n_frames)), np.array(mlx_model.encode(other, n_frames))
    )


def test_encode_zero_frames(models, parity_config):
    _, mlx_model = models
    assert mlx_model.encode(np.zeros((parity_config.n_mels, 16), np.float32), 0).shape == (0, parity_config.d_model)


def test_mlx_matches_torch_on_tiny_model(models, parity_config):
    # 1秒分の乱数melでencode → step(t=0, token=blank)のlogits
    torch_model, mlx_model = models
    mel = _mel(parity_config, 101)
    t_enc = torch_model.encode(mel, 101)
    m_enc = mlx_model.encode(mel, 101)
    t_logits, _ = torch_model.step(t_enc, 0, parity_config.blank_id, torch_model.init_state())
    m_logits, _ = mlx_model.step(m_enc, 0, parity_config.blank_id, mlx_model.init_state())
    assert m_logits.dtype == np.float32
    assert m_logits.shape == (parity_config.vocab_size + 1 + len(parity_config.durations),)
    np.testing.assert_allclose(m_logits, t_logits, rtol=RTOL, atol=ATOL)


def test_init_state_shape(models, parity_config):
    _, mlx_model = models
    rec, conv = mlx_model.init_state()
    shape = (parity_config.pred_layers, 1, parity_config.kda_state_width)
    assert rec.shape == conv.shape == shape
    assert rec.dtype == conv.dtype == mx.float32
    assert not np.array(rec).any() and not np.array(conv).any()


def test_mlx_step_state_roundtrip(models, parity_config):
    # 2ステップ進めた状態（rec・convの詰め方を含む）がtorchと一致する
    torch_model, mlx_model = models
    mel = _mel(parity_config, 48, seed=3)
    t_enc = torch_model.encode(mel, 48)
    m_enc = mlx_model.encode(mel, 48)
    t_state, m_state = torch_model.init_state(), mlx_model.init_state()
    for t, token in ((0, 3), (1, 7)):
        t_logits, t_state = torch_model.step(t_enc, t, token, t_state)
        m_logits, m_state = mlx_model.step(m_enc, t, token, m_state)
        np.testing.assert_allclose(m_logits, t_logits, rtol=RTOL, atol=ATOL)
    for got, expected in zip(m_state, t_state, strict=True):
        assert got.dtype == mx.float32
        np.testing.assert_allclose(np.array(got), expected.numpy(), rtol=RTOL, atol=ATOL)
    assert np.array(m_state[0]).any() and np.array(m_state[1]).any()  # 0のままの比較で空振りしない


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mlx_greedy_tokens_equal_torch_on_tiny(models, parity_config, seed):
    torch_model, mlx_model = models
    mel = _mel(parity_config, 203, seed=seed)
    expected = greedy_tdt(torch_model, torch_model.encode(mel, 203))
    got = greedy_tdt(mlx_model, mlx_model.encode(mel, 203))
    assert expected  # 空の列どうしの比較で空振りしない
    assert got == expected


def test_bf16_default_stays_close_to_float32(parity_mlx_dir, models, parity_config):
    # 既定（エンコーダbf16、正規化・softmax・KDA・jointはfloat32）でもfloat32の結果から大きく外れない
    _, f32 = models
    bf16 = HaasMLX.load(parity_mlx_dir)
    mel = _mel(parity_config, 101)
    enc = bf16.encode(mel, 101)
    assert enc.dtype == mx.bfloat16
    ref = np.array(f32.encode(mel, 101))
    err = np.abs(np.array(enc.astype(mx.float32)) - ref).max() / np.abs(ref).max()
    assert err < 3e-2
    logits, _ = bf16.step(enc, 0, parity_config.blank_id, bf16.init_state())
    ref_logits, _ = f32.step(f32.encode(mel, 101), 0, parity_config.blank_id, f32.init_state())
    assert logits.dtype == np.float32
    assert np.abs(logits - ref_logits).max() / np.abs(ref_logits).max() < 3e-2


def _current_cache_limit() -> int:
    """mlx 0.32にはget_cache_limitが無いので、set_cache_limitの戻り値で読み、同じ値に戻す。"""
    current = mx.set_cache_limit(0)
    mx.set_cache_limit(current)
    return current


@pytest.mark.parametrize(
    ("before", "requested", "expected"),
    [
        (1 << 40, 12345, 12345),  # 大きい上限は要求値まで下げる
        (1000, 1 << 30, 1000),  # 利用者が設定した小さい上限は上げない
        (1 << 40, None, 1 << 40),  # Noneなら触らない
    ],
)
def test_load_caps_metal_buffer_cache(parity_mlx_dir, before, requested, expected):
    previous = mx.set_cache_limit(before)
    try:
        HaasMLX.load(parity_mlx_dir, cache_limit_bytes=requested)
        assert _current_cache_limit() == expected
    finally:
        mx.set_cache_limit(previous)


def test_load_caps_cache_at_one_gib_by_default(parity_mlx_dir):
    previous = mx.set_cache_limit(1 << 40)
    try:
        HaasMLX.load(parity_mlx_dir)
        assert _current_cache_limit() <= 1 << 30
    finally:
        mx.set_cache_limit(previous)
