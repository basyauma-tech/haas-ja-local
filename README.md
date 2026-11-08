# haas-ja-local

[English below](#english)

**非公式の変換・推論パッケージです。** Third Intelligence, Inc.とは提携・承認・後援の関係にありません。

[Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)をNeMo・CUDA・Tritonなしでローカル実行します。MLX重みは元のbf16の値を保持し、軸の並べ替えと推論に不要な値の除去のみ。Hugging Face上の分類はquantizedですが、bf16のMLX版は量子化していません。

| 形式 | Hugging Face | 状態 |
|---|---|---|
| MLX（Apple Silicon、bf16） | [basyauma-tech/Haas-ja-v1.0-mlx](https://huggingface.co/basyauma-tech/Haas-ja-v1.0-mlx) | 公開中 |
| MLX（Apple Silicon、線形層8bit量子化） | basyauma-tech/Haas-ja-v1.0-mlx-8bit | 準備中 |
| ONNX（CPU、fp32・int8） | basyauma-tech/Haas-ja-v1.0-onnx | 準備中 |

精度・速度・制限: [MLXモデルカード](docs/model-cards/mlx/README.md)。

## 使い方

Python 3.12が必要です。

```sh
# MLX版で認識する（macOS、Apple Silicon）
pip install "haas-ja-local[mlx] @ git+https://github.com/basyauma-tech/haas-ja-local"
# ONNX版で認識する（macOS・Linux・Windows）
pip install "haas-ja-local[onnx] @ git+https://github.com/basyauma-tech/haas-ja-local"
# 元の.nemoから変換する（torchが入ります）
pip install "haas-ja-local[convert] @ git+https://github.com/basyauma-tech/haas-ja-local"
```

```sh
hf download basyauma-tech/Haas-ja-v1.0-mlx --local-dir Haas-ja-v1.0-mlx
```

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("Haas-ja-v1.0-mlx")
result = transcribe(backend, tokenizer, load_audio("audio.wav"))
print(result.text)
```

```sh
haas-ja-local transcribe Haas-ja-v1.0-mlx audio.wav
haas-ja-local transcribe Haas-ja-v1.0-mlx audio.wav --json --timestamps
```

- 30秒超は静かな位置で約30秒ごとに分割し、結果を連結。
- MLXのMetalキャッシュは既定1GiB。`HaasMLX.load`の`cache_limit_bytes`で変更（`None`は設定しない）。

MLXバックエンドは読み込み時に環境変数`MLX_ENABLE_TF32`を既定で0にし、予測ネットワークと結合ネットワークをfloat32で計算します。TF32を使う場合は読み込み前に`MLX_ENABLE_TF32=1`を設定してください。

モデルの自動判定:

- `model.safetensors`があればMLX。
- `fp32/`または`int8/`があればONNX。
- `.nemo`ならPyTorch（`[convert]`extraが必要）。

## 変換

```sh
hf download third-intelligence/Haas-ja-v1.0 Haas-ja-v1.0.nemo --revision 660728d39019e5e065046154beca44ac9a66a540 --local-dir upstream
haas-ja-local convert-mlx --nemo upstream/Haas-ja-v1.0.nemo --out Haas-ja-v1.0-mlx
haas-ja-local convert-onnx --nemo upstream/Haas-ja-v1.0.nemo --out Haas-ja-v1.0-onnx
```

`convert-mlx --quantize 8`は線形層を8bitに量子化した版を出力（mlxが必要なためmacOSのみ）。

`convert-onnx --int8`は動的量子化版を`int8/`にも出力（精度検証中）。`fp32/`があればfp32を使用。`transcribe --variant int8`で切り替えます。

公開MLX版の元`.nemo`のSHA-256: `c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa`。

## 検証

- [数値検証](docs/verification/2026-11-08-mac-numerics.md): 純PyTorchと第三者float32 ONNX。
- [MLX検証](docs/verification/2026-11-08-mac-mlx.md): asr-ja-bench v1.1のFLEURS・JVNVのCER、速度・メモリ・長時間音声。
- [MLX量子化版の検証](docs/verification/2026-11-08-mac-mlx-quant.md): 8bit・4bitのCER、速度・メモリ。

## ライセンス

Apache-2.0（[LICENSE](LICENSE)、[NOTICE](NOTICE)）。元モデル: Third Intelligence, Inc.（Apache-2.0）。

参考実装: NVIDIA NeMo、third-intelligence/Haas-ASR、matsuurajun/haas-mlx、Sanami/haas-ja-onnx（Apache-2.0）、fla-org/flash-linear-attention（MIT）。

---

# English

**This is an unofficial conversion/inference package.** It is not affiliated with, endorsed by, or sponsored by Third Intelligence, Inc.

Run [Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) locally without NeMo, CUDA or Triton. MLX preserves original bf16 values, only rearranging axes and removing unused values. Hugging Face classifies it as quantized, but the bf16 MLX version is not quantized.

| Format | Hugging Face | Status |
|---|---|---|
| MLX (Apple Silicon, bf16) | [basyauma-tech/Haas-ja-v1.0-mlx](https://huggingface.co/basyauma-tech/Haas-ja-v1.0-mlx) | published |
| MLX (Apple Silicon, 8-bit linear layers) | basyauma-tech/Haas-ja-v1.0-mlx-8bit | planned |
| ONNX (CPU, fp32 and int8) | basyauma-tech/Haas-ja-v1.0-onnx | planned |

Accuracy, speed and limitations: [MLX model card](docs/model-cards/mlx/README.md).

## Installation and usage

Requires Python 3.12.

```sh
pip install "haas-ja-local[mlx] @ git+https://github.com/basyauma-tech/haas-ja-local"
pip install "haas-ja-local[onnx] @ git+https://github.com/basyauma-tech/haas-ja-local"
pip install "haas-ja-local[convert] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download basyauma-tech/Haas-ja-v1.0-mlx --local-dir Haas-ja-v1.0-mlx
haas-ja-local transcribe Haas-ja-v1.0-mlx audio.wav
haas-ja-local transcribe Haas-ja-v1.0-mlx audio.wav --json --timestamps
```

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("Haas-ja-v1.0-mlx")
print(transcribe(backend, tokenizer, load_audio("audio.wav")).text)
```

Extras: `mlx` for macOS/Apple Silicon; `onnx` for macOS/Linux/Windows; `convert` installs torch.

- Audio above 30 seconds: split at quiet points every ~30 seconds; join results.
- MLX Metal cache: 1 GiB default. Set `cache_limit_bytes` in `HaasMLX.load` (`None` leaves it unchanged).

On loading, the MLX backend defaults the environment variable `MLX_ENABLE_TF32` to 0 and computes the prediction and joint networks in float32. To use TF32, set `MLX_ENABLE_TF32=1` before loading.

Model detection:

- `model.safetensors` present: MLX.
- `fp32/` or `int8/` present: ONNX.
- `.nemo`: PyTorch (requires the `[convert]` extra).

## Conversion

```sh
hf download third-intelligence/Haas-ja-v1.0 Haas-ja-v1.0.nemo --revision 660728d39019e5e065046154beca44ac9a66a540 --local-dir upstream
haas-ja-local convert-mlx --nemo upstream/Haas-ja-v1.0.nemo --out Haas-ja-v1.0-mlx
haas-ja-local convert-onnx --nemo upstream/Haas-ja-v1.0.nemo --out Haas-ja-v1.0-onnx
```

`convert-mlx --quantize 8` writes a version with 8-bit linear layers (needs mlx, so macOS only).

`convert-onnx --int8` also exports dynamic-quantized weights to `int8/` (accuracy under verification). Uses fp32 if `fp32/` exists; select int8 with `transcribe --variant int8`.

Published MLX source `.nemo` SHA-256: `c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa`.

## Verification

- [Numerical checks](docs/verification/2026-11-08-mac-numerics.md): pure PyTorch versus third-party float32 ONNX.
- [MLX checks](docs/verification/2026-11-08-mac-mlx.md): asr-ja-bench v1.1 FLEURS/JVNV CER, speed, memory, long audio.
- [MLX quantized checks](docs/verification/2026-11-08-mac-mlx-quant.md): 8-bit and 4-bit CER, speed, memory.

Records are in Japanese.

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Original model: Third Intelligence, Inc. (Apache-2.0).

References: NVIDIA NeMo, third-intelligence/Haas-ASR, matsuurajun/haas-mlx, Sanami/haas-ja-onnx (Apache-2.0); fla-org/flash-linear-attention (MIT).
