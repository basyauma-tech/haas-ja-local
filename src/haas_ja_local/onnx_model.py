"""ONNX Runtimeのバックエンド（CPU、推論専用、単一発話）。

convert-onnxが書いたディレクトリ（fp32/かint8/、config.json、preprocessor.npz）を読む。
エンコーダとデコーダ＋結合の入出力名はNeMoのRNNT書き出しと同じ。torchには依存しない。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover - onnxはextra
    raise ImportError("the onnx backend needs onnxruntime: pip install 'haas-ja-local[onnx]'") from e

from haas_ja_local.config import HaasConfig
from haas_ja_local.decode import Backend

VARIANTS = ("fp32", "int8")
ENCODER_FILE = "encoder.onnx"
DECODER_FILE = "decoder_joint.onnx"


def _session(path: Path, threads: int | None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class HaasONNX(Backend):
    """Backendの実装。encodeはnumpy [T',d_model] float32を返し、状態はnumpyの組(rec, conv)。"""

    def __init__(
        self,
        config: HaasConfig,
        encoder: ort.InferenceSession,
        decoder: ort.InferenceSession,
        window: np.ndarray,
        fb: np.ndarray,
        variant: str,
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.window_np = window
        self.fb_np = fb
        self.variant = variant

    @property
    def sessions(self) -> tuple[ort.InferenceSession, ort.InferenceSession]:
        return self.encoder, self.decoder

    @classmethod
    def load(cls, model_dir: Path, *, variant: str | None = None, threads: int | None = None) -> HaasONNX:
        """model_dir（config.json・preprocessor.npzとfp32/かint8/を含む）を読む。

        variant: "fp32"か"int8"。Noneならint8/があればint8、無ければfp32。
        threads: ONNX Runtimeのintra-opスレッド数。Noneなら既定（物理コア数）。
        """
        model_dir = Path(model_dir)
        if variant is None:
            variant = "int8" if (model_dir / "int8").is_dir() else "fp32"
        elif variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
        sub = model_dir / variant
        for path in (sub / ENCODER_FILE, sub / DECODER_FILE, model_dir / "config.json", model_dir / "preprocessor.npz"):
            if not path.is_file():
                raise FileNotFoundError(f"ONNX model file not found: {path}")

        config = HaasConfig.from_json(model_dir / "config.json")
        with np.load(model_dir / "preprocessor.npz") as npz:
            window = np.ascontiguousarray(npz["window"], dtype=np.float32)
            fb = np.ascontiguousarray(npz["fb"], dtype=np.float32)
        expected_fb = (config.n_mels, config.n_fft // 2 + 1)
        if window.shape != (config.win_length,) or fb.shape != expected_fb:
            raise ValueError(
                f"preprocessor.npz: window {window.shape} / fb {fb.shape} do not match "
                f"({config.win_length},) / {expected_fb}"
            )
        encoder = _session(sub / ENCODER_FILE, threads)
        decoder = _session(sub / DECODER_FILE, threads)
        return cls(config, encoder, decoder, window, fb, variant)

    def encode(self, mel: np.ndarray, n_frames: int) -> np.ndarray:
        if n_frames <= 0:
            return np.zeros((0, self.config.d_model), dtype=np.float32)
        outputs, lengths = self.encoder.run(
            None,
            {
                "audio_signal": np.ascontiguousarray(mel, dtype=np.float32)[None],
                "length": np.array([n_frames], dtype=np.int64),
            },
        )
        return np.ascontiguousarray(outputs[0, :, : int(lengths[0])].T, dtype=np.float32)

    def num_frames(self, encoded: np.ndarray) -> int:
        return int(encoded.shape[0])

    def init_state(self) -> tuple[np.ndarray, np.ndarray]:
        shape = (self.config.pred_layers, 1, self.config.kda_state_width)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    def step(
        self, encoded: np.ndarray, t: int, token: int, state: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        outputs, _, rec, conv = self.decoder.run(
            None,
            {
                "encoder_outputs": encoded[t][None, :, None],
                "targets": np.array([[token]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "input_states_1": state[0],
                "input_states_2": state[1],
            },
        )
        return outputs[0, 0, 0].astype(np.float32), (rec, conv)
