import numpy as np
import pytest
import torch

from haas_ja_local.features import log_mel, mel_frames


def _torch_reference(samples, window, fb, cfg):
    """NeMoのFilterbankFeatures.forward（推論時）の式をtorchでそのまま書いた参照実装。"""
    x = torch.from_numpy(samples)
    x = torch.cat([x[:1], x[1:] - 0.97 * x[:-1]])
    n_frames = (len(samples) + 2 * (cfg.n_fft // 2) - cfg.n_fft) // cfg.hop_length  # get_seq_len
    spec = torch.stft(
        x,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=torch.from_numpy(window),
        center=True,
        pad_mode="constant",
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


@pytest.mark.parametrize("n", [320, 400, 160 * 16, 160 * 16 + 1, 160 * 17 + 80, 16000 * 3 + 7])
def test_log_mel_matches_torch_reference(n, tiny_config):
    x, window, fb = _inputs(n, tiny_config)
    mel, n_frames = log_mel(x, window, fb, tiny_config)
    ref, ref_frames = _torch_reference(x, window, fb, tiny_config)
    assert n_frames == ref_frames == n // 160 == mel_frames(n, tiny_config)
    assert mel.dtype == np.float32 and mel.shape[0] == tiny_config.n_mels
    # 幅はSTFTのフレーム数（n_frames+1）を16の倍数に切り上げたもの。torchの参照と同じ
    assert mel.shape == ref.shape and mel.shape[1] == -(-(n_frames + 1) // 16) * 16
    np.testing.assert_allclose(mel[:, :n_frames], ref[:, :n_frames], rtol=1e-4, atol=1e-4)
    assert np.all(mel[:, n_frames:] == 0)


@pytest.mark.parametrize("n", [160, 200, 319])
def test_log_mel_single_frame_is_zero(n, tiny_config):
    # 160〜319サンプルは1フレーム。不偏標準偏差が定義できない（NeMoはNaNを0にする）ので、そのフレームは0になる
    _, window, fb = _inputs(n, tiny_config)
    mel, n_frames = log_mel(np.full(n, 0.3, np.float32), window, fb, tiny_config)
    assert n_frames == 1 == mel_frames(n, tiny_config)
    assert mel.shape == (tiny_config.n_mels, 16) and mel.dtype == np.float32
    assert np.all(mel == 0)


def test_log_mel_too_short_has_no_frames(tiny_config):
    # 160サンプル未満は0フレーム。呼び出し側が空として扱えるよう(n_mels, 0)を返す
    _, window, fb = _inputs(100, tiny_config)
    mel, n = log_mel(np.full(100, 0.3, np.float32), window, fb, tiny_config)
    assert n == 0 == mel_frames(100, tiny_config)
    assert mel.shape == (tiny_config.n_mels, 0) and mel.dtype == np.float32


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
