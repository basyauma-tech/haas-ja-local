import numpy as np
import pytest
import torch

from haas_ja_local.features import log_mel, mel_frames


def _torch_reference(samples, window, fb, cfg, pad_mode="reflect"):
    """NeMoのFilterbankFeatures.forward（推論時）の式をtorchでそのまま書いた参照実装。"""
    x = torch.from_numpy(samples)
    x = torch.cat([x[:1], x[1:] - 0.97 * x[:-1]])
    n_frames = len(samples) // cfg.hop_length + 1  # get_seq_len: (n + 2*(n_fft//2) - n_fft) // hop + 1
    if pad_mode == "reflect":
        spec = torch.stft(
            x,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=torch.from_numpy(window),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
    else:  # 短い入力ではreflectで詰められないので、0埋めしてからcenter=Falseで同じフレームを作る
        pad = cfg.n_fft // 2
        spec = torch.stft(
            torch.nn.functional.pad(x, (pad, pad)),
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=torch.from_numpy(window),
            center=False,
            return_complex=True,
        )
    power = spec.abs() ** 2
    mel = torch.log(torch.from_numpy(fb) @ power + 2**-24)
    valid = mel[:, :n_frames]
    mean = valid.mean(dim=1, keepdim=True)
    std = torch.std(valid, dim=1, unbiased=True, keepdim=True)
    mel = (mel - mean) / (std + 1e-5)
    mel[:, n_frames:] = 0
    remainder = mel.shape[1] % cfg.pad_to
    if remainder:
        mel = torch.nn.functional.pad(mel, (0, cfg.pad_to - remainder))
    return mel.numpy(), n_frames


def _inputs(n, cfg, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32) * 0.1
    window = np.hanning(cfg.win_length + 1)[:-1].astype(np.float32)  # 値は何でもよい（同じ窓を両方に渡す）
    fb = rng.random((cfg.n_mels, cfg.n_fft // 2 + 1), dtype=np.float32)
    return x, window, fb


@pytest.mark.parametrize("n", [160 * 16, 160 * 16 + 1, 160 * 17 + 80, 16000 * 3 + 7])
def test_log_mel_matches_torch_reference(n, tiny_config):
    x, window, fb = _inputs(n, tiny_config)
    mel, n_frames = log_mel(x, window, fb, tiny_config)
    ref, ref_frames = _torch_reference(x, window, fb, tiny_config)
    assert n_frames == ref_frames == n // 160 + 1 == mel_frames(n, tiny_config)
    assert mel.dtype == np.float32 and mel.shape[0] == tiny_config.n_mels
    assert mel.shape[1] % 16 == 0 and n_frames <= mel.shape[1] < n_frames + 16
    np.testing.assert_allclose(mel[:, :n_frames], ref[:, :n_frames], rtol=1e-4, atol=1e-4)
    assert np.all(mel[:, n_frames:] == 0)


def test_log_mel_short_input_uses_zero_padding(tiny_config):
    # 256サンプル以下はreflectで詰められない（NeMoは例外になる）ので0埋めにする
    x, window, fb = _inputs(200, tiny_config)
    mel, n_frames = log_mel(x, window, fb, tiny_config)
    ref, _ = _torch_reference(x, window, fb, tiny_config, pad_mode="constant")
    assert n_frames == 2 and mel.shape == (tiny_config.n_mels, 16)
    np.testing.assert_allclose(mel[:, :n_frames], ref[:, :n_frames], rtol=1e-4, atol=1e-4)
    assert np.all(mel[:, n_frames:] == 0)


def test_log_mel_too_short_returns_zero_frame(tiny_config):
    # 100サンプル: 100//160+1 = 1フレーム。不偏標準偏差が定義できないので、そのフレームは0にする
    _, window, fb = _inputs(100, tiny_config)
    mel, n = log_mel(np.full(100, 0.3, np.float32), window, fb, tiny_config)
    assert n == 1 and mel.shape == (tiny_config.n_mels, 16) and mel.dtype == np.float32
    assert np.all(mel == 0)


def test_log_mel_empty(tiny_config):
    _, window, fb = _inputs(0, tiny_config)
    mel, n = log_mel(np.zeros(0, np.float32), window, fb, tiny_config)
    assert n == 0 == mel_frames(0, tiny_config)
    assert mel.shape == (tiny_config.n_mels, 0) and mel.dtype == np.float32


def test_log_mel_rejects_wrong_shapes(tiny_config):
    x, window, fb = _inputs(1600, tiny_config)
    with pytest.raises(ValueError, match="fb"):
        log_mel(x, window, fb[None], tiny_config)  # checkpointの(1, n_mels, 257)をそのまま渡した場合
    with pytest.raises(ValueError, match="window"):
        log_mel(x, window[:-1], fb, tiny_config)
    with pytest.raises(ValueError, match="samples"):
        log_mel(x[None], window, fb, tiny_config)
