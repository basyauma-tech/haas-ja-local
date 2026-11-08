# haas-ja-local

## 概要

日本語音声認識モデル[Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)を、NeMoなしでローカル実行するための変換・推論パッケージです。Apple Silicon向けのMLX版と、CPU向けのONNX版を提供する予定です。

本パッケージは非公式の変換であり、Third Intelligence, Inc.とは無関係です。

## インストール

（後のタスクで記載します）

## 使い方

変換済みモデルは別途公開します（公開先は後日ここに記載します）。モデルのディレクトリか元の`.nemo`ファイルを指定します。

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("path/to/model")
result = transcribe(backend, tokenizer, load_audio("audio.wav"))
print(result.text)
```

```sh
haas-ja-local transcribe path/to/model audio.wav --json --timestamps
```

30秒を超える音声は無音に近い位置で約30秒ごとに分けて認識し、結果をつなげます。

## 精度

（後のタスクで記載します）

## ライセンス

Apache-2.0。詳細は[LICENSE](LICENSE)と[NOTICE](NOTICE)を参照してください。

---

## Overview

A NeMo-free conversion and inference package for running the Japanese ASR model
[Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) locally.
It is planned to provide an MLX backend for Apple Silicon and an ONNX backend for CPU.

This is an unofficial conversion and is not affiliated with Third Intelligence, Inc.

## Installation

(To be filled in by a later task.)

## Usage

Converted models are published separately (links will be added here later). Pass a model directory or the original `.nemo` file.

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("path/to/model")
result = transcribe(backend, tokenizer, load_audio("audio.wav"))
print(result.text)
```

```sh
haas-ja-local transcribe path/to/model audio.wav --json --timestamps
```

Audio longer than 30 seconds is split at quiet points into chunks of about 30 seconds, and the results are joined.

## Accuracy

(To be filled in by a later task.)

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
