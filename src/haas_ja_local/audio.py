"""音声ファイルの読み込み。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def load_audio(path: str | Path, *, sample_rate: int = 16000) -> np.ndarray:
    """音声ファイルを読み、sample_rateのモノラルfloat32（[-1,1]）の1次元配列にする。

    複数チャンネルは平均し、サンプリング周波数が違えばsoxrで変換する。
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    y = data.mean(axis=1, dtype=np.float32)
    if sr != sample_rate:
        y = soxr.resample(y, sr, sample_rate, quality="HQ")
    return np.ascontiguousarray(y, dtype=np.float32)
