import shutil

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")
pytest.importorskip("torch")

from haas_ja_local.architecture import encoded_length  # noqa: E402
from haas_ja_local.convert.onnx import export_onnx  # noqa: E402
from haas_ja_local.decode import greedy_tdt  # noqa: E402
from haas_ja_local.onnx_model import HaasONNX  # noqa: E402
from haas_ja_local.pipeline import load_model, transcribe  # noqa: E402
from haas_ja_local.torch_model import load_torch_model  # noqa: E402

RTOL = ATOL = 1e-4


@pytest.fixture(scope="module")
def parity_onnx_dir(parity_nemo, tmp_path_factory):
    out = tmp_path_factory.mktemp("parity-onnx")
    export_onnx(parity_nemo, out)
    return out


@pytest.fixture(scope="module")
def models(parity_nemo, parity_onnx_dir):
    """(torch float32, ONNX fp32)。"""
    torch_model, _ = load_torch_model(parity_nemo)
    return torch_model, HaasONNX.load(parity_onnx_dir, variant="fp32")


def _mel(config, n_frames: int, seed: int = 0) -> np.ndarray:
    padded = -(-n_frames // config.pad_to) * config.pad_to
    return np.random.default_rng(seed).standard_normal((config.n_mels, padded)).astype(np.float32)


def _fake_int8(bundle):
    """int8/の選択を見るだけのため、fp32をそのまま写す（量子化はtest_convert_onnxで見る）。"""
    shutil.copytree(bundle / "fp32", bundle / "int8")


def test_load_selects_variant(parity_onnx_dir, tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(parity_onnx_dir, bundle)
    assert HaasONNX.load(bundle).variant == "fp32"
    _fake_int8(bundle)
    assert HaasONNX.load(bundle).variant == "fp32"  # 両方あれば既定はfp32（int8のCERは未計測）
    assert HaasONNX.load(bundle, variant="int8").variant == "int8"
    with pytest.raises(ValueError, match="variant"):
        HaasONNX.load(bundle, variant="fp16")
    shutil.rmtree(bundle / "fp32")
    assert HaasONNX.load(bundle).variant == "int8"  # fp32/が無ければint8
    with pytest.raises(FileNotFoundError, match="fp32"):
        HaasONNX.load(bundle, variant="fp32")


def test_load_model_forwards_variant(parity_onnx_dir, tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(parity_onnx_dir, bundle)
    _fake_int8(bundle)
    assert load_model(bundle)[0].variant == "fp32"
    assert load_model(bundle, variant="int8")[0].variant == "int8"
    assert load_model(bundle, backend="onnx", variant="fp32")[0].variant == "fp32"


def test_load_without_model_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        HaasONNX.load(tmp_path)


def test_load_sets_threads_and_cpu_provider(parity_onnx_dir):
    model = HaasONNX.load(parity_onnx_dir, threads=2)
    for sess in model.sessions:
        assert sess.get_providers() == ["CPUExecutionProvider"]
        options = sess.get_session_options()
        assert options.intra_op_num_threads == 2
        assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL


def test_preprocessor_and_config_match_torch(models, parity_config):
    torch_model, onnx_model = models
    assert onnx_model.config == torch_model.config
    assert onnx_model.window_np.dtype == onnx_model.fb_np.dtype == np.float32
    np.testing.assert_array_equal(onnx_model.window_np, torch_model.window_np)
    np.testing.assert_array_equal(onnx_model.fb_np, torch_model.fb_np)
    assert onnx_model.fb_np.shape == (parity_config.n_mels, parity_config.n_fft // 2 + 1)


@pytest.mark.parametrize("n_frames", [101, 97, 33, 8, 1])
def test_encode_matches_torch(models, parity_config, n_frames):
    torch_model, onnx_model = models
    mel = _mel(parity_config, n_frames)
    got = onnx_model.encode(mel, n_frames)
    assert isinstance(got, np.ndarray) and got.dtype == np.float32 and got.flags.c_contiguous
    assert got.shape == (encoded_length(n_frames, parity_config), parity_config.d_model)
    assert onnx_model.num_frames(got) == got.shape[0]
    np.testing.assert_allclose(got, torch_model.encode(mel, n_frames).numpy(), rtol=RTOL, atol=ATOL)


def test_encode_zero_frames(models, parity_config):
    _, onnx_model = models
    got = onnx_model.encode(np.zeros((parity_config.n_mels, 16), np.float32), 0)
    assert got.shape == (0, parity_config.d_model)
    assert greedy_tdt(onnx_model, got) == []


def test_init_state(models, parity_config):
    _, onnx_model = models
    rec, conv = onnx_model.init_state()
    shape = (parity_config.pred_layers, 1, parity_config.kda_state_width)
    assert rec.shape == conv.shape == shape
    assert rec.dtype == conv.dtype == np.float32
    assert not rec.any() and not conv.any()


def test_step_matches_torch(models, parity_config):
    torch_model, onnx_model = models
    mel = _mel(parity_config, 48, seed=3)
    t_enc = torch_model.encode(mel, 48)
    o_enc = onnx_model.encode(mel, 48)
    t_state, o_state = torch_model.init_state(), onnx_model.init_state()
    for t, token in ((0, parity_config.blank_id), (1, 3), (2, 7)):
        t_logits, t_state = torch_model.step(t_enc, t, token, t_state)
        o_logits, o_state = onnx_model.step(o_enc, t, token, o_state)
        assert o_logits.dtype == np.float32
        assert o_logits.shape == (parity_config.vocab_size + 1 + len(parity_config.durations),)
        np.testing.assert_allclose(o_logits, t_logits, rtol=RTOL, atol=ATOL)
    for got, expected in zip(o_state, t_state, strict=True):
        np.testing.assert_allclose(got, expected.numpy(), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_greedy_tokens_equal_torch(models, parity_config, seed):
    torch_model, onnx_model = models
    mel = _mel(parity_config, 203, seed=seed)
    expected = greedy_tdt(torch_model, torch_model.encode(mel, 203))
    got = greedy_tdt(onnx_model, onnx_model.encode(mel, 203))
    assert expected  # 空の列どうしの比較で空振りしない
    assert got == expected


def test_load_model_dispatches_directory_to_onnx(parity_onnx_dir, parity_nemo, parity_config):
    backend, tokenizer = load_model(parity_onnx_dir, threads=1)
    assert isinstance(backend, HaasONNX)
    assert backend.sessions[0].get_session_options().intra_op_num_threads == 1
    assert tokenizer.vocab_size == parity_config.vocab_size
    audio = np.random.default_rng(0).standard_normal(16000 * 2).astype(np.float32) * 0.1
    torch_backend, torch_tokenizer = load_model(parity_nemo)
    expected = transcribe(torch_backend, torch_tokenizer, audio)
    got = transcribe(backend, tokenizer, audio)
    assert got.tokens == expected.tokens
    assert got.text == expected.text
