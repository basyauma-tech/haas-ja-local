import numpy as np
import soundfile as sf

from haas_ja_local.audio import load_audio


def test_load_resamples_and_downmixes(tmp_path):
    sr = 48000
    t = np.arange(sr) / sr
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.zeros_like(t)], axis=1).astype(np.float32)
    sf.write(tmp_path / "a.wav", stereo, sr, subtype="PCM_16")
    y = load_audio(tmp_path / "a.wav")
    assert y.dtype == np.float32 and y.ndim == 1 and abs(len(y) - 16000) <= 1
    assert y.flags["C_CONTIGUOUS"]
    assert 0.2 < np.abs(y).max() < 0.6  # 平均で振幅が半分になる


def test_load_flac_passthrough_16k(tmp_path):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000 + 123) * 0.1).astype(np.float32)
    sf.write(tmp_path / "a.flac", x, 16000, subtype="PCM_16")
    y = load_audio(str(tmp_path / "a.flac"))
    assert y.dtype == np.float32 and y.ndim == 1 and len(y) == len(x)
    np.testing.assert_allclose(y, x, atol=1 / 32768)  # PCM_16の量子化誤差まで一致
