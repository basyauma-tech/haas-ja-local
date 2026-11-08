---
license: apache-2.0
language:
  - ja
library_name: onnx
pipeline_tag: automatic-speech-recognition
base_model: third-intelligence/Haas-ja-v1.0
base_model_relation: quantized
tags:
  - onnx
  - onnxruntime
  - cpu
  - japanese
  - speech-recognition
  - asr
  - parakeet
  - tdt
  - fastconformer
---

<!-- 骨組み。数値と確認結果はTask 12（x86_64での検証）とTask 13（公開）で埋める。TODOが残っている間は公開しない。 -->

# Haas-ja-v1.0-onnx

[English version below](#english)

**非公式の変換物です。** 日本語音声認識モデル[third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)を、basyauma-technologyがONNX形式に変換しました。Third Intelligence, Inc.とは提携・承認・後援の関係にありません。fp32版は元のbf16の重みをfloat32にしてONNXに書き出したもので、int8版はそれをONNX Runtimeの動的量子化で8ビット整数にしたものです。Hugging Face上の分類はquantizedですが、量子化しているのはint8版だけで、fp32版は量子化していません。

## 概要

元のHaas-ja-v1.0は、FastConformerエンコーダとTDTデコーダからなる日本語音声認識モデルで、予測ネットワークにKimi Delta Attentionを使います。公式の実行環境はLinux＋NVIDIA GPU＋NeMo 2.7で、CPUとmacOSは公式の対象外です。

このリポジトリは、ONNX RuntimeのCPU実行でNeMo・CUDA・Tritonなしに動かすために、元の`.nemo`をONNXに書き出したものです。推論コードは同梱していません。GitHubの[basyauma-tech/haas-ja-local](https://github.com/basyauma-tech/haas-ja-local)をインストールして使います。

| ファイル | 内容 |
|---|---|
| `fp32/encoder.onnx`、`fp32/encoder.onnx.data` | エンコーダ（float32）。約2.4GBでprotobufの上限（2GB）を超えるため、重みを外部データファイルに分けている。TODO(Task 12/13): 実際のサイズ |
| `fp32/decoder_joint.onnx` | 予測ネットワークと結合ネットワーク（float32） |
| `fp32/SHA256SUMS` | `fp32/`の各ファイルのSHA-256 |
| `int8/encoder.onnx`、`int8/decoder_joint.onnx` | ONNX Runtimeの動的量子化（per-channel）。TODO(Task 12/13): 量子化の設定（`reduce_range`、デコーダをfp32に残したか）、`.data`の有無、サイズ |
| `int8/SHA256SUMS` | `int8/`の各ファイルのSHA-256 |
| `config.json` | 構成（MLX版と同じ） |
| `tokenizer.model` | 元の`.nemo`に含まれるSentencePieceモデル（BPE、3,072語彙）をそのまま |
| `preprocessor.npz` | log-mel用の窓とメルフィルタバンク（チェックポイントのbf16をfloat32にしたもの）。ONNXグラフには前処理を含めないため、別ファイルで配る |
| `LICENSE`、`NOTICE` | Apache-2.0の全文と帰属表示 |

TODO(Task 12/13): fp32とint8のどちらを既定にしたかと、その理由。

## 動作環境

- ONNX Runtimeが動くCPU（macOS・Linux・Windows）。
- Python 3.12（パッケージの要件は3.12以上3.13未満）。
- 実行時の依存はonnxruntime・numpy・sentencepiece・soundfile・soxrです。torchとNeMoは要りません。
- TODO(Task 12/13): 動作と計測を確かめた機種・CPU・OS・onnxruntimeの版。確かめていない環境。

## 使い方

インストールとダウンロード（`hf`コマンドは`huggingface_hub`に含まれます）:

```sh
pip install "haas-ja-local[onnx] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download basyauma-technology/Haas-ja-v1.0-onnx --local-dir Haas-ja-v1.0-onnx
```

Python:

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("Haas-ja-v1.0-onnx")
result = transcribe(backend, tokenizer, load_audio("audio.wav"))
print(result.text)
```

コマンドライン（`--threads`でONNX Runtimeのスレッド数を指定できます）:

```sh
haas-ja-local transcribe Haas-ja-v1.0-onnx audio.wav --threads 4
```

`int8/`があればint8を、無ければfp32を使います。fp32を使うには`fp32/`だけをダウンロードするか、`HaasONNX.load(path, variant="fp32")`で読みます。TODO(Task 12/13): この説明を確かめる。

長時間音声とタイムスタンプの扱いはMLX版と同じです（30秒を超える音声は約30秒ごとに分けて認識し、`--json --timestamps`でトークンごとの時刻を出します）。

## 入出力の仕様

NeMoのRNNT ONNX書き出しと同じ名前にしています。opsetは17です。

エンコーダ（`encoder.onnx`）:

| 名前 | 向き | 形 | 型 | 説明 |
|---|---|---|---|---|
| `audio_signal` | 入力 | [B,128,T] | float32 | log-mel。Tは16の倍数にパディングする |
| `length` | 入力 | [B] | int64 | パディング前のフレーム数 |
| `outputs` | 出力 | [B,1024,T'] | float32 | エンコーダ出力（80ms/フレーム） |
| `encoded_lengths` | 出力 | [B] | int64 | 有効なT' |

予測ネットワークと結合ネットワーク（`decoder_joint.onnx`）:

| 名前 | 向き | 形 | 型 | 説明 |
|---|---|---|---|---|
| `encoder_outputs` | 入力 | [B,1024,1] | float32 | エンコーダ出力の1フレーム |
| `targets` | 入力 | [B,1] | int32 | 直前のトークン（最初はblank） |
| `target_length` | 入力 | [B] | int32 | 常に1 |
| `input_states_1` | 入力 | [2,B,20480] | float32 | 層ごとのKDAの再帰状態（5×64×64） |
| `input_states_2` | 入力 | [2,B,20480] | float32 | 層ごとのq/k/vのshort convバッファ（3×320×4＝3,840を先頭に置き、残りは0） |
| `outputs` | 出力 | [B,1,1,3078] | float32 | logits（トークン3,072＋blank 1＋duration 5） |
| `prednet_lengths` | 出力 | [B] | int32 | `target_length`をそのまま返す |
| `output_states_1` | 出力 | [2,B,20480] | float32 | 更新後の`input_states_1` |
| `output_states_2` | 出力 | [2,B,20480] | float32 | 更新後の`input_states_2` |

- 状態の並び順は、[Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx)の公開スクリプト（`scripts/haas_onnx/decoder.py`）と同じです。
- Bは形の上ではバッチ次元ですが、配布する推論コードはバッチ1だけを扱います。
- log-melの計算はONNXグラフに含めません。`preprocessor.npz`の窓とフィルタバンクを使い、`haas_ja_local.features.log_mel`で計算します。
- ONNXの`metadata_props`に、sherpa-onnxのNeMo transducerの読み込みが参照するキー（`vocab_size`、`pred_rnn_layers`、`pred_hidden`、`model_type`、`normalize_type`など）と、変換元のSHA-256を入れています。TODO(Task 12/13): sherpa-onnxで実際に動いたかどうか（動かなければ「未確認」）。

## 精度

TODO(Task 12/13): asr-ja-bench v1.1のFLEURSとJVNVでの、fp32とint8のCER strict / flex / kana、公表値、差の表。測定条件（機種、CPU、OS、onnxruntimeの版、スレッド数、jacer v1.2.0）。未測定の一覧。torch（float32）との数値一致の結果。

## 速度とメモリ

TODO(Task 12/13): fp32とint8、スレッド数4と全コアでのRTFxとピークメモリの表。条件（バッチ1、1発話ずつ、CPU名、OS）。元モデルの公表RTFx（H200、バッチ512）とは比べられないことの注記。

## 変換方法

```sh
pip install "haas-ja-local[convert] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download third-intelligence/Haas-ja-v1.0 Haas-ja-v1.0.nemo --revision 660728d39019e5e065046154beca44ac9a66a540 --local-dir upstream
haas-ja-local convert-onnx --nemo upstream/Haas-ja-v1.0.nemo --out Haas-ja-v1.0-onnx --int8
```

| 項目 | 値 |
|---|---|
| 変換元 | [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)の`Haas-ja-v1.0.nemo` |
| リビジョン | `660728d39019e5e065046154beca44ac9a66a540`（2026-11-08取得） |
| SHA-256 | `c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa` |
| サイズ | 1,232,384,000バイト |
| 変換ツール | TODO(Task 12/13): haas-ja-localの版とコミット |

TODO(Task 12/13): 書き出しでしていること（float32への変換、トレース方式、int8の量子化設定）。

## 制限事項

- フィラー（「えー」「あの」など）は書き起こしから抜けやすく、表記（漢字・かな・数字）も指定できません。元モデルのモデルカードに記載のある性質です。
- 無音や雑音だけの区間で短い語が出る可能性があります。
- 30秒を超える音声を分けて認識すると、CERが悪化します。TODO(Task 12/13): ONNX版での数値、またはMLX版の記録への参照。
- ストリーミング認識には対応していません。
- 話者ラベルは出しません。
- 推論はバッチ1だけです。
- 元のHaas-ASRにあるカスタム語彙の機能はありません。
- タイムスタンプはトークン単位（80ms刻み）で、単語単位ではありません。

## ライセンスと帰属

- 重みと推論コードはApache-2.0です。重みは[third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)（Copyright Third Intelligence, Inc.、Apache-2.0）から変換したもので、帰属表示は`NOTICE`にあります。
- 「Haas」は元モデルの名称で、変換元を示すためだけに使っています。
- 関連するリンク: 元モデル[third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)、公式の推論パッケージ[third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR)、評価セット[third-intelligence/asr-ja-bench](https://huggingface.co/datasets/third-intelligence/asr-ja-bench)、CERの計算[third-intelligence/jacer](https://github.com/third-intelligence/jacer)。
- 評価データのライセンスは、FLEURSがCC BY 4.0、JVNVがCC BY-SA 4.0です。評価データはこのリポジトリに含めず、再配布もしていません。
- 実装にあたり、次の公開実装を参考にしました: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)、[third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR)、[matsuurajun/haas-mlx](https://github.com/matsuurajun/haas-mlx)、[Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx)（以上Apache-2.0）、[fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)（MIT）。Sanami/haas-ja-onnxのONNXは数値の比較にだけ使い、その重みは再配布していません。このリポジトリの重みは、すべて元の`.nemo`からbasyauma-technologyが変換したものです。

---

# English

**This is an unofficial conversion.** basyauma-technology converted the Japanese speech recognition model [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) to the ONNX format. It is not affiliated with, endorsed by, or sponsored by Third Intelligence, Inc. The fp32 variant is the original bf16 weights cast to float32 and exported to ONNX; the int8 variant is that export quantized to 8-bit integers with ONNX Runtime dynamic quantization. Hugging Face lists this repository under quantized; only the int8 variant is quantized, and the fp32 variant is not.

## Overview

Haas-ja-v1.0 is a Japanese speech recognition model with a FastConformer encoder and a TDT decoder whose prediction network uses Kimi Delta Attention. The official runtime is Linux with an NVIDIA GPU and NeMo 2.7; CPU and macOS are not officially supported.

This repository contains the original `.nemo` exported to ONNX, so that the model runs on CPU with ONNX Runtime, without NeMo, CUDA or Triton. It does not include inference code: install [basyauma-tech/haas-ja-local](https://github.com/basyauma-tech/haas-ja-local) from GitHub.

| File | Contents |
|---|---|
| `fp32/encoder.onnx`, `fp32/encoder.onnx.data` | Encoder (float32). About 2.4 GB, above the 2 GB protobuf limit, so the weights are stored in an external data file. TODO(Task 12/13): actual size |
| `fp32/decoder_joint.onnx` | Prediction and joint networks (float32) |
| `fp32/SHA256SUMS` | SHA-256 of each file in `fp32/` |
| `int8/encoder.onnx`, `int8/decoder_joint.onnx` | ONNX Runtime dynamic quantization (per-channel). TODO(Task 12/13): quantization settings (`reduce_range`, whether the decoder stays in fp32), whether a `.data` file exists, sizes |
| `int8/SHA256SUMS` | SHA-256 of each file in `int8/` |
| `config.json` | Configuration (same as the MLX version) |
| `tokenizer.model` | The SentencePiece model (BPE, 3,072 tokens) from the original `.nemo`, unchanged |
| `preprocessor.npz` | Window and mel filter bank for the log-mel features (the checkpoint's bf16 values as float32). The ONNX graphs do not include preprocessing, so these are shipped separately |
| `LICENSE`, `NOTICE` | Full Apache-2.0 text and attribution |

TODO(Task 12/13): which of fp32 and int8 is the default, and why.

## Requirements

- Any CPU that ONNX Runtime supports (macOS, Linux, Windows).
- Python 3.12 (the package requires >=3.12,<3.13).
- Runtime dependencies: onnxruntime, numpy, sentencepiece, soundfile and soxr. Neither torch nor NeMo is needed.
- TODO(Task 12/13): machine, CPU, OS and onnxruntime version used for testing and measurement; untested environments.

## Usage

Install and download (the `hf` command comes with `huggingface_hub`):

```sh
pip install "haas-ja-local[onnx] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download basyauma-technology/Haas-ja-v1.0-onnx --local-dir Haas-ja-v1.0-onnx
```

Python:

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("Haas-ja-v1.0-onnx")
result = transcribe(backend, tokenizer, load_audio("audio.wav"))
print(result.text)
```

Command line (`--threads` sets the number of ONNX Runtime threads):

```sh
haas-ja-local transcribe Haas-ja-v1.0-onnx audio.wav --threads 4
```

int8 is used when `int8/` exists, otherwise fp32. To use fp32, download only `fp32/` or load it with `HaasONNX.load(path, variant="fp32")`. TODO(Task 12/13): confirm this description.

Long audio and timestamps work as in the MLX version (audio longer than 30 seconds is split into pieces of about 30 seconds, and `--json --timestamps` prints per-token times).

## Input and output specification

Names follow NeMo's RNNT ONNX export. opset 17.

Encoder (`encoder.onnx`):

| Name | Direction | Shape | Type | Description |
|---|---|---|---|---|
| `audio_signal` | input | [B,128,T] | float32 | log-mel; T is padded to a multiple of 16 |
| `length` | input | [B] | int64 | number of frames before padding |
| `outputs` | output | [B,1024,T'] | float32 | encoder output (80 ms per frame) |
| `encoded_lengths` | output | [B] | int64 | valid T' |

Prediction and joint networks (`decoder_joint.onnx`):

| Name | Direction | Shape | Type | Description |
|---|---|---|---|---|
| `encoder_outputs` | input | [B,1024,1] | float32 | one encoder frame |
| `targets` | input | [B,1] | int32 | previous token (blank at the start) |
| `target_length` | input | [B] | int32 | always 1 |
| `input_states_1` | input | [2,B,20480] | float32 | per-layer KDA recurrent state (5×64×64) |
| `input_states_2` | input | [2,B,20480] | float32 | per-layer q/k/v short-conv buffers (3×320×4 = 3,840 values first, the rest zero) |
| `outputs` | output | [B,1,1,3078] | float32 | logits (3,072 tokens + 1 blank + 5 durations) |
| `prednet_lengths` | output | [B] | int32 | `target_length` passed through |
| `output_states_1` | output | [2,B,20480] | float32 | updated `input_states_1` |
| `output_states_2` | output | [2,B,20480] | float32 | updated `input_states_2` |

- The state layout matches the public scripts of [Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx) (`scripts/haas_onnx/decoder.py`).
- B is a batch dimension in the graph, but the inference code handles batch size 1 only.
- The ONNX graphs do not compute log-mel features. Use the window and filter bank in `preprocessor.npz` with `haas_ja_local.features.log_mel`.
- The ONNX `metadata_props` include the keys that sherpa-onnx's NeMo transducer loader reads (`vocab_size`, `pred_rnn_layers`, `pred_hidden`, `model_type`, `normalize_type` and others) and the SHA-256 of the source. TODO(Task 12/13): whether it actually runs in sherpa-onnx ("not checked" if not).

## Accuracy

TODO(Task 12/13): table of CER strict / flex / kana for fp32 and int8 on FLEURS and JVNV of asr-ja-bench v1.1, with published values and differences. Conditions (machine, CPU, OS, onnxruntime version, threads, jacer v1.2.0). List of what was not measured. Numerical agreement with torch (float32).

## Speed and memory

TODO(Task 12/13): table of RTFx and peak memory for fp32 and int8 with 4 threads and all cores. Conditions (batch size 1, one utterance at a time, CPU model, OS). Note that the published RTFx of the original model (H200, batch size 512) is not comparable.

## Conversion

```sh
pip install "haas-ja-local[convert] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download third-intelligence/Haas-ja-v1.0 Haas-ja-v1.0.nemo --revision 660728d39019e5e065046154beca44ac9a66a540 --local-dir upstream
haas-ja-local convert-onnx --nemo upstream/Haas-ja-v1.0.nemo --out Haas-ja-v1.0-onnx --int8
```

| Item | Value |
|---|---|
| Source | `Haas-ja-v1.0.nemo` from [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) |
| Revision | `660728d39019e5e065046154beca44ac9a66a540` (downloaded 2026-11-08) |
| SHA-256 | `c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa` |
| Size | 1,232,384,000 bytes |
| Converter | TODO(Task 12/13): haas-ja-local version and commit |

TODO(Task 12/13): what the export does (cast to float32, tracing method, int8 quantization settings).

## Limitations

- Fillers (such as "えー" and "あの") tend to be omitted, and the writing style (kanji, kana, numerals) cannot be chosen. The original model card describes this behaviour.
- Short words may appear in stretches of silence or noise.
- Splitting audio longer than 30 seconds raises the CER. TODO(Task 12/13): numbers for the ONNX version, or a reference to the MLX record.
- No streaming recognition.
- No speaker labels.
- Batch size 1 only.
- No custom vocabulary, which the official Haas-ASR package offers.
- Timestamps are per token (in 80 ms steps), not per word.

## License and attribution

- The weights and the inference code are released under Apache-2.0. The weights are converted from [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) (Copyright Third Intelligence, Inc., Apache-2.0); the attribution notice is in `NOTICE`.
- "Haas" is the name of the original model and is used here only to identify the source.
- Related links: the original model [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0), the official inference package [third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR), the benchmark [third-intelligence/asr-ja-bench](https://huggingface.co/datasets/third-intelligence/asr-ja-bench), and the CER library [third-intelligence/jacer](https://github.com/third-intelligence/jacer).
- Evaluation data licences: FLEURS is CC BY 4.0 and JVNV is CC BY-SA 4.0. The evaluation data is not included in or redistributed from this repository.
- The implementation consulted these public implementations: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo), [third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR), [matsuurajun/haas-mlx](https://github.com/matsuurajun/haas-mlx) and [Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx) (Apache-2.0), and [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT). The ONNX files from Sanami/haas-ja-onnx were used only for numerical comparison, and their weights are not redistributed. All weights in this repository were converted from the original `.nemo` by basyauma-technology.
