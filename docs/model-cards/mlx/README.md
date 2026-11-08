---
license: apache-2.0
language:
  - ja
library_name: mlx
pipeline_tag: automatic-speech-recognition
base_model: third-intelligence/Haas-ja-v1.0
base_model_relation: quantized
tags:
  - mlx
  - apple-silicon
  - japanese
  - speech-recognition
  - asr
  - parakeet
  - tdt
  - fastconformer
---

# Haas-ja-v1.0-mlx

[English version below](#english)

**非公式の変換物です。** 日本語音声認識モデル[third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)を、basyauma-technologyがMLX形式に変換しました。Third Intelligence, Inc.とは提携・承認・後援の関係にありません。重みは元のbf16の値のままで、MLXの配置に合わせてテンソルの軸を並べ替えただけです。

## 概要

元のHaas-ja-v1.0は、FastConformerエンコーダとTDTデコーダからなる日本語音声認識モデルで、予測ネットワークにKimi Delta Attentionを使います。配布物は`Haas-ja-v1.0.nemo`の1ファイルで、公式の実行環境はLinux＋NVIDIA GPU＋NeMo 2.7です。予測ネットワークがTritonとCUDAを必要とするため、CPUとmacOSは公式の対象外です。

このリポジトリは、Apple SiliconのMac上でNeMo・CUDA・Tritonなしに動かすために、元の`.nemo`をMLX用の重みに変換したものです。推論コードは同梱していません。GitHubの[basyauma-tech/haas-ja-local](https://github.com/basyauma-tech/haas-ja-local)をインストールして使います。

| ファイル | 内容 |
|---|---|
| `model.safetensors` | 重み（bfloat16、1,231,827,296バイト）。元のチェックポイントの値とキー名を変えず、畳み込みの重みの軸だけをMLXの配置に並べ替えた |
| `config.json` | 構成（寸法、語彙数、durationの候補、変換元の`.nemo`のSHA-256） |
| `tokenizer.model` | 元の`.nemo`に含まれるSentencePieceモデル（BPE、3,072語彙）をそのまま |
| `LICENSE`、`NOTICE` | Apache-2.0の全文と帰属表示 |

`model.safetensors`のSHA-256は`96600ed4ca8a4d9d45e6ed6bfd276ab9109da3f1c9e20e2bb0870e6f894fe5e4`です。

## 動作環境

- Apple SiliconのMac（MLXはmacOSでだけ動きます）。
- Python 3.12（パッケージの要件は3.12以上3.13未満）。
- 実行時の依存はmlx・numpy・sentencepiece・soundfile・soxrです。torchとNeMoは要りません。
- 重みのダウンロードに約1.2GBの空き容量が要ります。
- 動作と計測を確かめたのはApple M5 Max（メモリ128GB）、macOS 27.2、mlx 0.32.4の1台だけです。Intel Mac、メモリ16GB以下の機種では確かめていません。

## 使い方

インストールとダウンロード（`hf`コマンドは`huggingface_hub`に含まれます。無ければ`pip install huggingface_hub`で入れます）:

```sh
pip install "haas-ja-local[mlx] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download basyauma-technology/Haas-ja-v1.0-mlx --local-dir Haas-ja-v1.0-mlx
```

Python:

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("Haas-ja-v1.0-mlx")
result = transcribe(backend, tokenizer, load_audio("audio.wav"))
print(result.text)
```

コマンドライン:

```sh
haas-ja-local transcribe Haas-ja-v1.0-mlx audio.wav
```

音声はsoundfileで読める形式（WAV・FLACなど）を受け付けます。複数チャンネルは平均してモノラルにし、16kHz以外はsoxrで16kHzに変換します。

長時間音声: 30秒を超える音声は自動で分けて認識します。約30秒ごとに、目標位置の前後2秒の範囲で最も静かな0.1秒の窓の中心で切り、断片を1つずつ順に認識して結果をつなぎます。断片はまとめて処理しません。MLX版はMetalのバッファキャッシュを既定で1GiBに抑えます（`HaasMLX.load`の`cache_limit_bytes`で変更でき、`None`なら設定しません）。

タイムスタンプ: `--json --timestamps`を付けると、ファイルごとに1行のJSONを出し、トークンごとの開始・終了秒を含めます。時刻はエンコーダのフレーム（80ms）単位です。

```sh
haas-ja-local transcribe Haas-ja-v1.0-mlx audio.wav --json --timestamps
```

```json
{"file": "audio.wav", "text": "今日は良い天気ですね。", "tokens": [{"text": " 今", "start": 0.0, "end": 0.32}, {"text": "日は", "start": 0.32, "end": 0.64}, {"text": "良", "start": 0.64, "end": 0.88}, {"text": "い", "start": 0.88, "end": 1.04}, {"text": "天", "start": 1.04, "end": 1.28}, {"text": "気", "start": 1.28, "end": 1.6}, {"text": "ですね", "start": 1.6, "end": 1.84}, {"text": "。", "start": 1.84, "end": 2.0}]}
```

## 精度

[asr-ja-bench](https://huggingface.co/datasets/third-intelligence/asr-ja-bench) v1.1のFLEURSとJVNVで、CERを[jacer](https://github.com/third-intelligence/jacer) v1.2.0で測りました。値は%です。

| サブセット | 発話数 | 音声（時間） | 指標 | この変換 | 公表値 | 差（ポイント） |
|---|---:|---:|---|---:|---:|---:|
| FLEURS | 650 | 2.36 | CER strict | 3.65 | 3.65〜3.67 | 0.00 |
| FLEURS | 650 | 2.36 | CER flex | 2.98 | 未照合 | - |
| FLEURS | 650 | 2.36 | CER kana | 2.33 | 未照合 | - |
| JVNV | 1,375 | 3.33 | CER strict | 4.51 | 4.50〜4.53 | 0.00 |
| JVNV | 1,375 | 3.33 | CER flex | 3.06 | 未照合 | - |
| JVNV | 1,375 | 3.33 | CER kana | 2.25 | 未照合 | - |

測定条件:

- 評価データは、データセット付属の`rebuild.py`で元のデータから再構築しました。書き起こしはすべて`rebuild.py`のハッシュ照合を通りました。JVNVの音声は、`rebuild.py`が前提とするffmpeg 7.1ではなくffmpeg 9.0.2で16kHzに変換したため、公式の評価音声とわずかに違う可能性があります。
- 1発話ずつ認識し、サブセット全体で`jacer.cer`を1回計算しました（コーパス単位のマイクロ平均）。jacerは3指標とも句読点と空白を採点しません。30秒を超える発話はなく、長時間音声の分割は使っていません。
- 実行環境はApple M5 Max、macOS 27.2、mlx 0.32.4（Metal）、バッチ1です。エンコーダはbf16、予測ネットワークと結合ネットワークはfloat32で計算します。
- 公表値は、元モデルのモデルカードと公開記事にあるasr-ja-bench v1.1での公式の測定（NeMo、bf16、H200）です。両者で小数第2位が違うため幅で示しました。差は幅の近い端からの差で、幅の中なら0としました。flexとkanaは公表値と照合していません。
- 精度を合わせるための調整はしていません。1回目の実行の数値です。

実装の確認: 推論コードの基準となる純PyTorch実装を、同じチェックポイントから第三者が書き出したfloat32のONNXとFLEURSの6発話で比べました。log-melの最大絶対差は9.54e-07、エンコーダ出力の相対誤差は最大4.36e-06、結合ネットワークのlogitsの相対誤差は最大1.16e-06で、トークン列は6発話すべてで一致しました。

未測定:

- asr-ja-benchの残りのサブセット（JSUT basic5000、TEDxJP-10K、SpeechBSD）。そのため、5サブセットの重み付きCERは出していません。
- Intel Mac、メモリ16GB以下の機種。
- バッチ2以上（推論コードはバッチ1だけを扱います）。
- 同じ話者が続けて話す実際の長時間録音。

計測の詳細と再現手順は、GitHubの[検証記録（精度と速度）](https://github.com/basyauma-tech/haas-ja-local/blob/main/docs/verification/2026-11-08-mac-mlx.md)と[検証記録（数値一致）](https://github.com/basyauma-tech/haas-ja-local/blob/main/docs/verification/2026-11-08-mac-numerics.md)にあります。

## 速度とメモリ

FLEURSの650発話のうち、先頭3発話をウォームアップとして除いた647発話（合計2.35時間）で測りました。条件はバッチ1、1発話ずつ、Metal、エンコーダbf16、予測ネットワークと結合ネットワークfloat32です。処理時間は音声ファイルの読み込みを除いた認識だけの時間です。GBは10^9バイトです。

| 項目 | 既定（キャッシュ上限1GiB） | 上限なし（参考） |
|---|---:|---:|
| RTFx（1発話ごとの中央値） | 254.2 | 250.9 |
| RTFx（合計音声÷合計処理時間） | 256.1 | 255.9 |
| ピークRSS | 1.57GB | 2.68GB |
| mlxのピーク割り当て（`mx.get_peak_memory()`） | 1.76GB | 1.82GB |
| macOSのピークメモリフットプリント | 3.19GB | 14.40GB |

- 1発話の処理時間の中央値は0.050秒、モデルの読み込みは0.15秒でした（どちらも上限なしでの測定。読み込みはファイルがOSのキャッシュに載った状態）。
- 33.8分（2,026.5秒）の音声1本をCLIで認識すると、プロセスの起動とモデルの読み込みを含めて6.97秒、最大RSSは1.74GB、ピークメモリフットプリントは3.48GBでした。分割数は68です。
- 元モデルが公表しているRTFxは、NVIDIA H200でバッチ512の測定です。条件がまったく違うため、この表とは比べられません。
- キャッシュに上限を設けないと、mlxが解放したMetalバッファを持ち続け、33.8分の音声でメモリフットプリントが13.96GBまで増えました。実際に使っているメモリは1.25GBで一定で、リークではありません。このため既定で上限を設けています。

## 変換方法

このリポジトリの重みは、次の手順で元の`.nemo`から作りました。

```sh
pip install "haas-ja-local[convert] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download third-intelligence/Haas-ja-v1.0 Haas-ja-v1.0.nemo --revision 660728d39019e5e065046154beca44ac9a66a540 --local-dir upstream
haas-ja-local convert-mlx --nemo upstream/Haas-ja-v1.0.nemo --out Haas-ja-v1.0-mlx
```

| 項目 | 値 |
|---|---|
| 変換元 | [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)の`Haas-ja-v1.0.nemo` |
| リビジョン | `660728d39019e5e065046154beca44ac9a66a540`（2026-11-08取得） |
| SHA-256 | `c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa` |
| サイズ | 1,232,384,000バイト |
| 変換ツール | haas-ja-local 0.1.0（`model.safetensors`のメタデータに記録） |

変換でしていること:

- `.nemo`（tar）から`model_weights.ckpt`と`*_tokenizer.model`をtorchで読みます。NeMoは使いません。
- キー名と形状が元のモデルの構成と一致することを確かめてから書き出します。
- 値はbf16のまま変えません。量子化も型変換もしていません。
- 畳み込みの重みだけ、軸をMLXの配置（チャンネルが最後）に並べ替えます。2次元は[O,I,H,W]を[O,H,W,I]に、1次元（深さ方向の畳み込みを含む）は[O,I,K]を[O,K,I]にします。
- 推論に使わない`num_batches_tracked`は捨てます。
- `tokenizer.model`はそのまま書き出し、`config.json`は元の構成から必要な値を書き出します。

同じ`.nemo`から変換し直すと、テンソルの値はビット単位で同じになります。ただし、safetensorsがヘッダ内のメタデータのキーの順序を固定しないため、ファイル全体のSHA-256は上の値と一致しないことがあります。

## 制限事項

- フィラー（「えー」「あの」など）は書き起こしから抜けやすく、表記（漢字・かな・数字）も指定できません。元モデルのモデルカードに記載のある性質です。
- 無音や雑音だけの区間で短い語が出ることがありえます。10秒の無音と弱いホワイトノイズの合成音声4種では文字は出ませんでしたが、実際の録音では確かめていません。
- 30秒を超える音声を分けて認識すると、CERが悪化します。FLEURSの150発話をつないだ33.8分の音声で、1発話ずつ認識してつなぐとCER strictは4.10%、30秒前後で分けて認識すると6.44%でした。PyTorch（float32）でも6.39%なので、MLXの数値誤差ではなく、モデルと分割のしかた（公式の`transcribe_longform`にならった規則）によるものです。67か所の境界のうち50か所が発話の途中にありました。話者の違う発話をつないだ音声での結果なので、実際の録音では差が小さい可能性があります。
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
- 実装にあたり、次の公開実装（いずれもApache-2.0）を参考にしました: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)、[third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR)、[fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)、[matsuurajun/haas-mlx](https://github.com/matsuurajun/haas-mlx)、[Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx)。Sanami/haas-ja-onnxのfloat32のONNXは数値の比較にだけ使い、その重みは再配布していません。このリポジトリの重みは、すべて元の`.nemo`から自分で変換したものです。

---

# English

**This is an unofficial conversion.** basyauma-technology converted the Japanese speech recognition model [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) to the MLX format. It is not affiliated with, endorsed by, or sponsored by Third Intelligence, Inc. The weights are the original bf16 values; only tensor axes were rearranged for MLX.

## Overview

Haas-ja-v1.0 is a Japanese speech recognition model with a FastConformer encoder and a TDT decoder whose prediction network uses Kimi Delta Attention. It is distributed as a single `Haas-ja-v1.0.nemo` file, and the official runtime is Linux with an NVIDIA GPU and NeMo 2.7. Its prediction network needs Triton and CUDA, so CPU and macOS are not officially supported.

This repository contains the original `.nemo` converted to MLX weights, so that the model runs on Apple Silicon Macs without NeMo, CUDA or Triton. It does not include inference code: install [basyauma-tech/haas-ja-local](https://github.com/basyauma-tech/haas-ja-local) from GitHub.

| File | Contents |
|---|---|
| `model.safetensors` | Weights (bfloat16, 1,231,827,296 bytes). Values and key names are those of the original checkpoint; only the axes of convolution weights are rearranged for MLX |
| `config.json` | Configuration (dimensions, vocabulary size, duration choices, SHA-256 of the source `.nemo`) |
| `tokenizer.model` | The SentencePiece model (BPE, 3,072 tokens) from the original `.nemo`, unchanged |
| `LICENSE`, `NOTICE` | Full Apache-2.0 text and attribution |

SHA-256 of `model.safetensors`: `96600ed4ca8a4d9d45e6ed6bfd276ab9109da3f1c9e20e2bb0870e6f894fe5e4`.

## Requirements

- A Mac with Apple Silicon (MLX runs on macOS only).
- Python 3.12 (the package requires >=3.12,<3.13).
- Runtime dependencies: mlx, numpy, sentencepiece, soundfile and soxr. Neither torch nor NeMo is needed.
- About 1.2 GB of disk space for the weights.
- Tested and measured on one machine only: Apple M5 Max (128 GB), macOS 27.2, mlx 0.32.4. Intel Macs and machines with 16 GB of memory or less have not been tested.

## Usage

Install and download (the `hf` command comes with `huggingface_hub`; install it with `pip install huggingface_hub` if needed):

```sh
pip install "haas-ja-local[mlx] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download basyauma-technology/Haas-ja-v1.0-mlx --local-dir Haas-ja-v1.0-mlx
```

Python:

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("Haas-ja-v1.0-mlx")
result = transcribe(backend, tokenizer, load_audio("audio.wav"))
print(result.text)
```

Command line:

```sh
haas-ja-local transcribe Haas-ja-v1.0-mlx audio.wav
```

Any format soundfile can read (WAV, FLAC and others) is accepted. Multiple channels are averaged to mono, and audio is resampled to 16 kHz with soxr when needed.

Long audio: audio longer than 30 seconds is split automatically. Roughly every 30 seconds, it is cut at the centre of the quietest 0.1-second window within 2 seconds of the target point; the pieces are transcribed one at a time and the results are joined. Pieces are not batched. The MLX backend caps the Metal buffer cache at 1 GiB by default (change it with `cache_limit_bytes` in `HaasMLX.load`; `None` leaves it untouched).

Timestamps: with `--json --timestamps`, one JSON line is printed per file, including start and end seconds for each token. Times are in units of encoder frames (80 ms).

```sh
haas-ja-local transcribe Haas-ja-v1.0-mlx audio.wav --json --timestamps
```

```json
{"file": "audio.wav", "text": "今日は良い天気ですね。", "tokens": [{"text": " 今", "start": 0.0, "end": 0.32}, {"text": "日は", "start": 0.32, "end": 0.64}, {"text": "良", "start": 0.64, "end": 0.88}, {"text": "い", "start": 0.88, "end": 1.04}, {"text": "天", "start": 1.04, "end": 1.28}, {"text": "気", "start": 1.28, "end": 1.6}, {"text": "ですね", "start": 1.6, "end": 1.84}, {"text": "。", "start": 1.84, "end": 2.0}]}
```

## Accuracy

CER was measured on the FLEURS and JVNV subsets of [asr-ja-bench](https://huggingface.co/datasets/third-intelligence/asr-ja-bench) v1.1 with [jacer](https://github.com/third-intelligence/jacer) v1.2.0. Values are in %.

| Subset | Utterances | Audio (hours) | Metric | This conversion | Published | Difference (points) |
|---|---:|---:|---|---:|---:|---:|
| FLEURS | 650 | 2.36 | CER strict | 3.65 | 3.65–3.67 | 0.00 |
| FLEURS | 650 | 2.36 | CER flex | 2.98 | not compared | - |
| FLEURS | 650 | 2.36 | CER kana | 2.33 | not compared | - |
| JVNV | 1,375 | 3.33 | CER strict | 4.51 | 4.50–4.53 | 0.00 |
| JVNV | 1,375 | 3.33 | CER flex | 3.06 | not compared | - |
| JVNV | 1,375 | 3.33 | CER kana | 2.25 | not compared | - |

Conditions:

- The evaluation data was rebuilt from the original datasets with the `rebuild.py` script shipped with the benchmark, and all transcripts passed its hash check. The JVNV audio was resampled to 16 kHz with ffmpeg 9.0.2 rather than the ffmpeg 7.1 that `rebuild.py` expects, so it may differ slightly from the official evaluation audio.
- Each utterance was transcribed on its own, and `jacer.cer` was computed once over each subset (corpus-level micro-average). jacer ignores punctuation and spaces in all three metrics. No utterance was longer than 30 seconds, so long-audio splitting was not used.
- Runtime: Apple M5 Max, macOS 27.2, mlx 0.32.4 (Metal), batch size 1. The encoder runs in bf16; the prediction and joint networks run in float32.
- Published values are the official asr-ja-bench v1.1 results (NeMo, bf16, H200) from the original model card and the release blog post. They differ in the second decimal place, so both are shown as a range. The difference is measured from the nearer end of the range and is 0 inside it. flex and kana were not compared with published values.
- Nothing was tuned to match the published values. These are the numbers from the first run.

Implementation check: the pure PyTorch implementation that serves as the reference for the inference code was compared, on six FLEURS utterances, with a float32 ONNX export of the same checkpoint made by a third party. The maximum absolute log-mel difference was 9.54e-07, the maximum relative error of the encoder output 4.36e-06 and that of the joint network logits 1.16e-06; the token sequences matched for all six utterances.

Not measured:

- The other asr-ja-bench subsets (JSUT basic5000, TEDxJP-10K, SpeechBSD), so no weighted CER over the five subsets is given.
- Intel Macs and machines with 16 GB of memory or less.
- Batch sizes above 1 (the inference code handles batch size 1 only).
- Real long recordings in which the same speaker keeps talking.

Details and reproduction steps are in the verification records on GitHub: [accuracy and speed](https://github.com/basyauma-tech/haas-ja-local/blob/main/docs/verification/2026-11-08-mac-mlx.md) and [numerical agreement](https://github.com/basyauma-tech/haas-ja-local/blob/main/docs/verification/2026-11-08-mac-numerics.md) (in Japanese).

## Speed and memory

Measured on 647 of the 650 FLEURS utterances (2.35 hours in total), excluding the first three as warm-up. Conditions: batch size 1, one utterance at a time, Metal, bf16 encoder, float32 prediction and joint networks. Times cover transcription only, not reading the audio file. GB means 10^9 bytes.

| Item | Default (1 GiB cache cap) | No cap (reference) |
|---|---:|---:|
| RTFx (median per utterance) | 254.2 | 250.9 |
| RTFx (total audio / total processing time) | 256.1 | 255.9 |
| Peak RSS | 1.57 GB | 2.68 GB |
| mlx peak allocation (`mx.get_peak_memory()`) | 1.76 GB | 1.82 GB |
| macOS peak memory footprint | 3.19 GB | 14.40 GB |

- The median time per utterance was 0.050 seconds and model loading took 0.15 seconds (both measured without the cap; loading with the files already in the OS cache).
- Transcribing one 33.8-minute file (2,026.5 seconds) with the CLI took 6.97 seconds including process start-up and model loading, with a maximum RSS of 1.74 GB and a peak memory footprint of 3.48 GB. The audio was split into 68 pieces.
- The RTFx published for the original model was measured on an NVIDIA H200 at batch size 512. The conditions are entirely different, so it cannot be compared with this table.
- Without a cache cap, mlx keeps freed Metal buffers, and the memory footprint grew to 13.96 GB on the 33.8-minute file. Memory actually in use stayed at 1.25 GB, so this is not a leak. That is why a cap is set by default.

## Conversion

The weights in this repository were produced from the original `.nemo` as follows.

```sh
pip install "haas-ja-local[convert] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download third-intelligence/Haas-ja-v1.0 Haas-ja-v1.0.nemo --revision 660728d39019e5e065046154beca44ac9a66a540 --local-dir upstream
haas-ja-local convert-mlx --nemo upstream/Haas-ja-v1.0.nemo --out Haas-ja-v1.0-mlx
```

| Item | Value |
|---|---|
| Source | `Haas-ja-v1.0.nemo` from [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) |
| Revision | `660728d39019e5e065046154beca44ac9a66a540` (downloaded 2026-11-08) |
| SHA-256 | `c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa` |
| Size | 1,232,384,000 bytes |
| Converter | haas-ja-local 0.1.0 (recorded in the `model.safetensors` metadata) |

What the conversion does:

- Reads `model_weights.ckpt` and `*_tokenizer.model` from the `.nemo` (a tar archive) with torch. NeMo is not used.
- Checks that the key names and shapes match the original model configuration before writing anything.
- Keeps the values in bf16 unchanged. Nothing is quantized or cast.
- Rearranges the axes of convolution weights only, to MLX's channels-last layout: 2-D weights from [O,I,H,W] to [O,H,W,I], 1-D weights (including depthwise convolutions) from [O,I,K] to [O,K,I].
- Drops `num_batches_tracked`, which inference does not use.
- Writes `tokenizer.model` unchanged and writes the needed values of the original configuration to `config.json`.

Converting the same `.nemo` again gives bit-identical tensor values. The SHA-256 of the whole file may still differ from the value above, because safetensors does not fix the order of metadata keys in the header.

## Limitations

- Fillers (such as "えー" and "あの") tend to be omitted, and the writing style (kanji, kana, numerals) cannot be chosen. The original model card describes this behaviour.
- Short words may appear in stretches of silence or noise. Four synthetic 10-second inputs (silence and weak white noise) produced no text, but this has not been checked on real recordings.
- Splitting audio longer than 30 seconds raises the CER. On a 33.8-minute file made by joining 150 FLEURS utterances, CER strict was 4.10% when each utterance was transcribed on its own and the results joined, and 6.44% when the file was split into pieces of about 30 seconds. PyTorch (float32) gives 6.39%, so this comes from the model and the way the audio is split (a rule modelled on the official `transcribe_longform`), not from MLX numerics. 50 of the 67 cut points fell inside an utterance. The file joins utterances from different speakers, so the gap may be smaller on real recordings.
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
- The implementation consulted these public implementations, all under Apache-2.0: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo), [third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR), [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention), [matsuurajun/haas-mlx](https://github.com/matsuurajun/haas-mlx) and [Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx). The float32 ONNX from Sanami/haas-ja-onnx was used only for numerical comparison, and its weights are not redistributed. All weights in this repository were converted from the original `.nemo` by us.
