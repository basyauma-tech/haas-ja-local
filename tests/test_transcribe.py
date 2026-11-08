import itertools
import sys

import numpy as np
import pytest

import haas_ja_local
from haas_ja_local.decode import TimedToken
from haas_ja_local.features import mel_frames
from haas_ja_local.pipeline import Transcript, load_model, transcribe

SR = 16000


@pytest.fixture(scope="module")
def tiny(tiny_nemo):
    return load_model(tiny_nemo)


class CountingBackend:
    """encodeの呼び出しごとに入力長を記録する代理。他の属性は元のバックエンドへ渡す。"""

    def __init__(self, inner):
        self._inner = inner
        self.encoded_frames: list[int] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def encode(self, mel, n_frames):
        self.encoded_frames.append(n_frames)
        return self._inner.encode(mel, n_frames)


def test_load_model_detects_torch_for_nemo(tiny_nemo):
    from haas_ja_local.torch_model import HaasTorch

    backend, tokenizer = load_model(tiny_nemo)
    assert isinstance(backend, HaasTorch)
    assert tokenizer.vocab_size == 16


def test_load_model_missing_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.nemo")


def test_load_model_unknown_directory_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"model\.safetensors"):
        load_model(tmp_path)


@pytest.mark.parametrize(
    ("layout", "module", "extra"),
    [
        ("model.safetensors", "mlx", "mlx"),
        ("fp32", "onnxruntime", "onnx"),
        ("int8", "onnxruntime", "onnx"),
    ],
)
def test_load_model_missing_backend_message(tmp_path, monkeypatch, layout, module, extra):
    if layout.endswith(".safetensors"):
        (tmp_path / layout).write_bytes(b"")
    else:
        (tmp_path / layout).mkdir()
    monkeypatch.setitem(sys.modules, module, None)  # importをImportErrorにする
    with pytest.raises(ImportError, match=rf"pip install 'haas-ja-local\[{extra}\]'"):
        load_model(tmp_path)


def test_load_model_missing_torch_message(tiny_nemo, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "haas_ja_local.torch_model", raising=False)
    with pytest.raises(ImportError, match=r"pip install 'haas-ja-local\[convert\]'"):
        load_model(tiny_nemo)


def test_load_model_explicit_backend_overrides_detection(tiny_nemo, monkeypatch):
    # .nemoでもbackend="onnx"ならonnxを読みに行く（ここではonnxruntimeが無い扱いで止める）
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match=r"haas-ja-local\[onnx\]"):
        load_model(tiny_nemo, backend="onnx")


def test_load_model_rejects_unknown_backend(tiny_nemo):
    with pytest.raises(ValueError, match="backend"):
        load_model(tiny_nemo, backend="tensorflow")


def test_load_model_rejects_variant_for_other_backends(tiny_nemo):
    with pytest.raises(ValueError, match="variant"):
        load_model(tiny_nemo, variant="fp32")
    with pytest.raises(ValueError, match="variant"):
        load_model(tiny_nemo, backend="torch", variant="int8")


def test_transcribe_too_short_returns_empty(tiny):
    backend, tokenizer = tiny
    result = transcribe(backend, tokenizer, np.zeros(100, np.float32))
    assert result == Transcript("", (), ())


def test_transcribe_silence_does_not_raise(tiny):
    backend, tokenizer = tiny
    result = transcribe(backend, tokenizer, np.zeros(3 * SR, np.float32))
    assert isinstance(result, Transcript)
    assert len(result.tokens) == len(result.offsets)
    assert all(o == 0.0 for o in result.offsets)


def test_transcribe_short_audio_is_one_pass(tiny):
    backend, tokenizer = tiny
    counting = CountingBackend(backend)
    audio = np.random.default_rng(0).standard_normal(int(SR * 3.5)).astype(np.float32)
    transcribe(counting, tokenizer, audio)
    assert counting.encoded_frames == [mel_frames(int(SR * 3.5), backend.config)]


def test_longform_offsets_increase(tiny):
    backend, tokenizer = tiny
    counting = CountingBackend(backend)
    # 3.5秒の乱数に窓長（0.1秒）の無音を1.0・2.0・3.0秒から置く。閾値1秒では各無音窓の中心
    # 1.05・2.05・3.05秒で切れて4断片になり、断片ごとにencodeする
    audio = np.random.default_rng(0).standard_normal(int(SR * 3.5)).astype(np.float32)
    for second in (1, 2, 3):
        audio[second * SR : second * SR + 1600] = 0.0
    result = transcribe(counting, tokenizer, audio, longform_threshold_seconds=1.0)
    cuts = [0, SR + 800, 2 * SR + 800, 3 * SR + 800, len(audio)]
    assert counting.encoded_frames == [mel_frames(b - a, backend.config) for a, b in itertools.pairwise(cuts)]
    assert len(result.offsets) == len(result.tokens)
    assert list(result.offsets) == sorted(result.offsets)
    assert set(result.offsets) <= {c / SR for c in cuts[:-1]}


class SpacedTokenizer:
    """idを文字にし、前後に空白を付けて返すトークナイザ（stripの有無を見分けるため）。"""

    def decode(self, ids):
        return " " + "".join(chr(ord("a") + i) for i in ids) + " "


def test_longform_joins_stripped_chunk_texts_and_offsets(tiny, monkeypatch):
    backend, _ = tiny
    # 分割とgreedyを差し替え、断片ごとの固定トークンでテキストの連結とoffsetの付け方を確かめる
    spans = [(0, SR), (SR, 2 * SR), (2 * SR, 3 * SR), (3 * SR, int(3.5 * SR))]
    calls = []

    def fake_split(audio, **kwargs):
        calls.append((len(audio), kwargs))
        return spans

    pieces = iter([[TimedToken(3, 0, 1)], [], [TimedToken(5, 2, 1), TimedToken(6, 4, 0)], [TimedToken(7, 1, 2)]])
    monkeypatch.setattr("haas_ja_local.pipeline.split_at_quiet_points", fake_split)
    monkeypatch.setattr("haas_ja_local.pipeline.greedy_tdt", lambda b, e: next(pieces))
    audio = np.random.default_rng(0).standard_normal(int(SR * 3.5)).astype(np.float32)
    result = transcribe(backend, SpacedTokenizer(), audio, longform_threshold_seconds=1.0)
    assert calls == [(int(3.5 * SR), {"sample_rate": SR, "chunk_seconds": 1.0})]
    assert [t.id for t in result.tokens] == [3, 5, 6, 7]
    assert result.offsets == (0.0, 2.0, 2.0, 3.0)
    assert result.text == "dfgh"


def test_package_exports_public_api():
    for name in ("__version__", "HaasConfig", "TimedToken", "Transcript", "load_audio", "load_model", "transcribe"):
        assert hasattr(haas_ja_local, name), name


def test_importing_package_does_not_import_heavy_backends():
    import subprocess

    code = (
        "import sys, haas_ja_local\n"
        "heavy = [m for m in ('torch', 'mlx', 'onnxruntime') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
