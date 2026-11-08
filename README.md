# haas-ja-local

[English below](#english)

## 概要

日本語音声認識モデル[Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)を、NeMo・CUDA・Tritonなしでローカル実行するための変換・推論パッケージです。元の`.nemo`から、Apple Silicon向けのMLX版とCPU向けのONNX版を作り、それぞれで認識できます。

本パッケージは非公式の変換・推論パッケージであり、Third Intelligence, Inc.とは提携・承認・後援の関係にありません。MLX版の重みは元のbf16の値のままで、MLXの配置に合わせてテンソルの軸を並べ替えただけです。

## 変換済みモデル

| 形式 | Hugging Face | 状態 |
|---|---|---|
| MLX（Apple Silicon、bf16） | [basyauma-technology/Haas-ja-v1.0-mlx](https://huggingface.co/basyauma-technology/Haas-ja-v1.0-mlx) | 公開準備中 |
| ONNX（CPU、fp32・int8） | basyauma-technology/Haas-ja-v1.0-onnx | 準備中 |

精度・速度・制限事項は各モデルカードに書きます。MLX版のカードの原稿は[docs/model-cards/mlx/README.md](docs/model-cards/mlx/README.md)です。

## インストール

Python 3.12が必要です。用途に応じてextraを選びます。

```sh
# MLX版で認識する（macOS、Apple Silicon）
pip install "haas-ja-local[mlx] @ git+https://github.com/basyauma-tech/haas-ja-local"
# ONNX版で認識する（macOS・Linux・Windows）
pip install "haas-ja-local[onnx] @ git+https://github.com/basyauma-tech/haas-ja-local"
# 元の.nemoから変換する（torchが入ります）
pip install "haas-ja-local[convert] @ git+https://github.com/basyauma-tech/haas-ja-local"
```

## 使い方

変換済みモデルのディレクトリか、元の`.nemo`ファイルを指定します。ディレクトリに`model.safetensors`があればMLX版、`fp32/`か`int8/`があればONNX版、`.nemo`ならPyTorch版で動きます。

```sh
hf download basyauma-technology/Haas-ja-v1.0-mlx --local-dir Haas-ja-v1.0-mlx
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

30秒を超える音声は無音に近い位置で約30秒ごとに分けて認識し、結果をつなげます。MLX版はMetalのバッファキャッシュを既定で1GiBに抑えます（`HaasMLX.load`の`cache_limit_bytes`で変更でき、`None`なら設定しません）。

## 自分で変換する

元の`.nemo`を取得し、MLX版かONNX版に変換します。

```sh
hf download third-intelligence/Haas-ja-v1.0 Haas-ja-v1.0.nemo --revision 660728d39019e5e065046154beca44ac9a66a540 --local-dir upstream
haas-ja-local convert-mlx --nemo upstream/Haas-ja-v1.0.nemo --out Haas-ja-v1.0-mlx
haas-ja-local convert-onnx --nemo upstream/Haas-ja-v1.0.nemo --out Haas-ja-v1.0-onnx --int8
```

公開しているMLX版は、SHA-256が`c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa`の`.nemo`から作りました。

## 検証記録

- [実モデルの数値検証（Mac）](docs/verification/2026-11-08-mac-numerics.md): 純PyTorch実装と第三者のfloat32 ONNXとの比較。
- [MLX版の精度と速度（Mac）](docs/verification/2026-11-08-mac-mlx.md): asr-ja-bench v1.1のFLEURSとJVNVでのCER、速度、メモリ、長時間音声。

## 開発

```sh
uv sync --extra mlx --extra onnx --extra convert --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## ライセンス

Apache-2.0。詳細は[LICENSE](LICENSE)と[NOTICE](NOTICE)を参照してください。元モデルはThird Intelligence, Inc.によるもので、Apache-2.0で公開されています。実装にあたり、NVIDIA NeMo、third-intelligence/Haas-ASR、fla-org/flash-linear-attention、matsuurajun/haas-mlx、Sanami/haas-ja-onnx（いずれもApache-2.0）を参考にしました。

---

# English

## Overview

A conversion and inference package for running the Japanese speech recognition model
[Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) locally without NeMo, CUDA or Triton.
It converts the original `.nemo` to an MLX version for Apple Silicon and an ONNX version for CPU, and transcribes with either.

This is an unofficial conversion and inference package. It is not affiliated with, endorsed by, or sponsored by
Third Intelligence, Inc. The MLX weights are the original bf16 values; only tensor axes were rearranged for MLX.

## Converted models

| Format | Hugging Face | Status |
|---|---|---|
| MLX (Apple Silicon, bf16) | [basyauma-technology/Haas-ja-v1.0-mlx](https://huggingface.co/basyauma-technology/Haas-ja-v1.0-mlx) | coming soon |
| ONNX (CPU, fp32 and int8) | basyauma-technology/Haas-ja-v1.0-onnx | coming soon |

Accuracy, speed and limitations are documented in each model card. The draft of the MLX card is
[docs/model-cards/mlx/README.md](docs/model-cards/mlx/README.md).

## Installation

Python 3.12 is required. Pick the extra for what you need.

```sh
# Transcribe with the MLX version (macOS, Apple Silicon)
pip install "haas-ja-local[mlx] @ git+https://github.com/basyauma-tech/haas-ja-local"
# Transcribe with the ONNX version (macOS, Linux, Windows)
pip install "haas-ja-local[onnx] @ git+https://github.com/basyauma-tech/haas-ja-local"
# Convert from the original .nemo (installs torch)
pip install "haas-ja-local[convert] @ git+https://github.com/basyauma-tech/haas-ja-local"
```

## Usage

Pass a converted model directory or the original `.nemo` file. A directory with `model.safetensors` runs on MLX,
one with `fp32/` or `int8/` runs on ONNX Runtime, and a `.nemo` file runs on PyTorch.

```sh
hf download basyauma-technology/Haas-ja-v1.0-mlx --local-dir Haas-ja-v1.0-mlx
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

Audio longer than 30 seconds is split at quiet points into chunks of about 30 seconds, and the results are joined. The MLX backend caps the Metal buffer cache at 1 GiB by default (change it with `cache_limit_bytes` in `HaasMLX.load`; `None` leaves it untouched).

## Converting it yourself

Download the original `.nemo` and convert it to the MLX or ONNX version.

```sh
hf download third-intelligence/Haas-ja-v1.0 Haas-ja-v1.0.nemo --revision 660728d39019e5e065046154beca44ac9a66a540 --local-dir upstream
haas-ja-local convert-mlx --nemo upstream/Haas-ja-v1.0.nemo --out Haas-ja-v1.0-mlx
haas-ja-local convert-onnx --nemo upstream/Haas-ja-v1.0.nemo --out Haas-ja-v1.0-onnx --int8
```

The published MLX version was converted from the `.nemo` with SHA-256
`c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa`.

## Verification records (in Japanese)

- [Numerical check of the real model (Mac)](docs/verification/2026-11-08-mac-numerics.md): the pure PyTorch implementation compared with a third-party float32 ONNX export.
- [Accuracy and speed of the MLX version (Mac)](docs/verification/2026-11-08-mac-mlx.md): CER on FLEURS and JVNV of asr-ja-bench v1.1, speed, memory and long audio.

## Development

```sh
uv sync --extra mlx --extra onnx --extra convert --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The original model is by Third Intelligence, Inc. and is
released under Apache-2.0. The implementation consulted NVIDIA NeMo, third-intelligence/Haas-ASR,
fla-org/flash-linear-attention, matsuurajun/haas-mlx and Sanami/haas-ja-onnx (all Apache-2.0).
