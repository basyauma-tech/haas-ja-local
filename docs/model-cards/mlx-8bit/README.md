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
  - 8-bit
  - quantized
---

# Haas-ja-v1.0-mlx-8bit

[English version below](#english)

**非公式の変換物です。** [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)の線形層を8bit量子化したMLX版です。Third Intelligence, Inc.とは提携・承認・後援の関係にありません。

重みは約0.73GBで、[bf16版](https://huggingface.co/basyauma-tech/Haas-ja-v1.0-mlx)の約0.59倍。ピークメモリは減りますが、検証機での速度はbf16版より約14%遅く、速くはなりません。

## 概要・環境

- Apple SiliconのMac専用。NeMo・CUDA・Triton・torch不要。Python 3.12と[haas-ja-local](https://github.com/basyauma-tech/haas-ja-local)を使用。
- 検証環境はbf16版と同じApple M5 Max（128GB）、macOS 27.2、mlx 0.32.4。他機種は未測定。

## 使い方

bf16版と同じコードで動きます。モデルのパスだけ変更します。`hf`がなければ`pip install huggingface_hub`で導入してください。

```sh
pip install "haas-ja-local[mlx] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download basyauma-tech/Haas-ja-v1.0-mlx-8bit --local-dir Haas-ja-v1.0-mlx-8bit
```

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("Haas-ja-v1.0-mlx-8bit")
result = transcribe(backend, tokenizer, load_audio("audio.wav"))
print(result.text)
```

```sh
haas-ja-local transcribe Haas-ja-v1.0-mlx-8bit audio.wav
haas-ja-local transcribe Haas-ja-v1.0-mlx-8bit audio.wav --json --timestamps
```

## 精度

asr-ja-bench v1.1をbf16版と同じ手順で評価。CER strictの単位は%、差はポイントです。bf16との差は表の丸め値から算出しています。

| サブセット | 発話数 | bf16 | 8bit | bf16との差 | 公表値 | 公表値との差 |
|---|---:|---:|---:|---:|---:|---:|
| FLEURS | 650 | 3.65 | 3.68 | +0.03 | 3.65〜3.67 | +0.01 |
| JVNV | 1,375 | 4.51 | 4.49 | -0.02 | 4.50〜4.53 | -0.01 |

公表値との差は範囲の近い端から計算し、範囲内なら0。8bitは合格基準の差0.3ポイント以内です。精度調整はしていません。

**4bit版は精度基準を外れたため公開していません。** CER strictはFLEURS3.70%（公表値との差+0.03ポイント）、JVNV5.00%（+0.47ポイント）でした。

詳細・再現手順: [量子化版の検証記録](https://github.com/basyauma-tech/haas-ja-local/blob/main/docs/verification/2026-11-08-mac-mlx-quant.md)。

## 実装の確認

小さな乱数モデルで変換・読み込みの往復、キーと形状、量子化しない値のビット一致、CLI引数を検証済み。逆量子化した重みをfloat32で持つモデルとの相対誤差は1e-5未満でした。実モデルとPyTorch（float32）のトークン列一致は未検証です。

## 速度とメモリ

上記M5 Max、バッチ1、逐次認識。FLEURSの先頭3発話をウォームアップとして除いた647発話で集計。キャッシュ上限は既定の1GiB。bf16は比較のため同じ時間帯に再測定した値です。GB＝10^9バイト。

| 項目 | bf16 | 8bit |
|---|---:|---:|
| RTFx（1発話ごとの中央値） | 260.1 | 223.3 |
| RTFx（合計音声÷合計処理時間） | 263.4 | 226.6 |
| 1発話の処理時間（中央値） | 0.048秒 | 0.057秒 |
| ピークRSS | 1.57GB | 1.09GB |
| mlxのピーク割り当て | 1.79GB | 1.23GB |
| macOSのピークメモリフットプリント | 3.20GB | 2.63GB |

## 変換方法

すべての線形層に`mlx.nn.quantize`の8bitアフィン量子化（グループサイズ64）を適用。畳み込み・正規化・埋め込みはbf16のままです。他の非量子化値もbf16版とビット単位で同じです。

```sh
haas-ja-local convert-mlx --nemo upstream/Haas-ja-v1.0.nemo --out Haas-ja-v1.0-mlx-8bit --quantize 8
```

変換環境と元の`.nemo`の取得手順は[bf16版の変換方法](https://huggingface.co/basyauma-tech/Haas-ja-v1.0-mlx#変換方法)を参照。読み込み時の`scales`・`biases`はエンコーダがbf16、予測・結合ネットワークがfloat32です。

## ファイル一覧

`var/dist/publish/mlx-8bit/`の実ファイルを`ls -l`と`shasum -a 256`で確認。サイズはバイトです。

| ファイル | サイズ | SHA-256 |
|---|---:|---|
| `model.safetensors` | 727,881,324 | `17ec1d12a67b35303c1bbc4defa3077b03f14665550e45d2749ada12ef73368e` |
| `config.json` | 726 | `0b7ad196af39a5120570d31bdcb0c9194e89d216892c9ea8e4d552170e8df6b3` |
| `tokenizer.model` | 279,711 | `9bd7bcdff62e43200001cd4c258d22e475b7b3b1e0f64f8be8e6e046f402f38c` |
| `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `NOTICE` | 741 | `cb6145131e3cb48e0bb432aa5b949ae72330ef0dddaf1621de9b0cc65ab8b51b` |

## 制限事項

[bf16版と同じ制限事項](https://huggingface.co/basyauma-tech/Haas-ja-v1.0-mlx#制限事項)が適用されます。量子化版では長時間音声の分割認識、無音・雑音入力、JSUT・TEDxJP-10K・SpeechBSD、グループサイズ64以外、メモリの少ない機種、バッチ2以上は未測定です。

## ライセンスと帰属

- bf16版と同じApache-2.0（重み・推論コード）。元重みは[third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)（Copyright Third Intelligence, Inc.）。帰属は`NOTICE`。「Haas」は変換元を示す名称。
- 参考実装: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)、[third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR)（公式推論パッケージ）、[matsuurajun/haas-mlx](https://github.com/matsuurajun/haas-mlx)、[Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx)はApache-2.0。[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)はMIT。
- Sanamiのfloat32 ONNXは数値比較のみで、重みは再配布しません。
- 評価データ: FLEURSはCC BY 4.0、JVNVはCC BY-SA 4.0。同梱・再配布しません。

---

# English

**This is an unofficial conversion** of [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) to MLX with 8-bit linear layers. It is not affiliated with, endorsed by, or sponsored by Third Intelligence, Inc.

Weights occupy about 0.73 GB, approximately 0.59 times the [bf16 version](https://huggingface.co/basyauma-tech/Haas-ja-v1.0-mlx). Peak memory decreases, but measured speed is about 14% lower than bf16; quantization does not make inference faster on the tested machine.

## Overview and requirements

- Apple Silicon Macs only; no NeMo, CUDA, Triton or torch. Requires Python 3.12 and [haas-ja-local](https://github.com/basyauma-tech/haas-ja-local).
- Same test environment as bf16: Apple M5 Max (128 GB), macOS 27.2, mlx 0.32.4. Other machines are unmeasured.

## Usage

Uses the same code as bf16; change only the model path. Install `hf` with `pip install huggingface_hub` if needed.

```sh
pip install "haas-ja-local[mlx] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download basyauma-tech/Haas-ja-v1.0-mlx-8bit --local-dir Haas-ja-v1.0-mlx-8bit
```

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("Haas-ja-v1.0-mlx-8bit")
result = transcribe(backend, tokenizer, load_audio("audio.wav"))
print(result.text)
```

```sh
haas-ja-local transcribe Haas-ja-v1.0-mlx-8bit audio.wav
haas-ja-local transcribe Haas-ja-v1.0-mlx-8bit audio.wav --json --timestamps
```

## Accuracy

asr-ja-bench v1.1, evaluated with the same procedure as bf16. CER strict values are percentages; differences are percentage points. Differences from bf16 use the rounded values shown.

| Subset | Utterances | bf16 | 8-bit | Difference from bf16 | Published | Difference from published |
|---|---:|---:|---:|---:|---:|---:|
| FLEURS | 650 | 3.65 | 3.68 | +0.03 | 3.65–3.67 | +0.01 |
| JVNV | 1,375 | 4.51 | 4.49 | -0.02 | 4.50–4.53 | -0.01 |

Differences from published values use the nearer endpoint, or 0 within the range. The 8-bit version passes the 0.3-point threshold. No accuracy tuning was applied.

**The 4-bit version is not published because it failed the accuracy threshold.** CER strict was 3.70% on FLEURS (+0.03 points from published) and 5.00% on JVNV (+0.47 points).

Details/reproduction (Japanese): [quantization verification](https://github.com/basyauma-tech/haas-ja-local/blob/main/docs/verification/2026-11-08-mac-mlx-quant.md).

## Implementation verification

Small random-model tests cover conversion/loading round trips, keys, shapes, bit-identical unquantized values and CLI arguments. Relative error against a model holding dequantized weights in float32 was below 1e-5. Token agreement between the full quantized model and PyTorch (float32) is unverified.

## Speed and memory

M5 Max setup above, batch 1, sequential recognition. 647 FLEURS utterances after excluding the first 3 warm-ups; default 1 GiB cache cap. bf16 was remeasured in the same time window. GB = 10^9 bytes.

| Item | bf16 | 8-bit |
|---|---:|---:|
| RTFx (median per utterance) | 260.1 | 223.3 |
| RTFx (total audio / total processing time) | 263.4 | 226.6 |
| Median processing time per utterance | 0.048 s | 0.057 s |
| Peak RSS | 1.57 GB | 1.09 GB |
| mlx peak allocation | 1.79 GB | 1.23 GB |
| macOS peak memory footprint | 3.20 GB | 2.63 GB |

## Conversion

All linear layers use `mlx.nn.quantize` with 8-bit affine quantization and group size 64. Convolutions, normalization and embeddings remain bf16. Other unquantized values are also bit-identical to bf16.

```sh
haas-ja-local convert-mlx --nemo upstream/Haas-ja-v1.0.nemo --out Haas-ja-v1.0-mlx-8bit --quantize 8
```

For conversion setup and source `.nemo` download, see [bf16 conversion](https://huggingface.co/basyauma-tech/Haas-ja-v1.0-mlx#conversion). On loading, `scales` and `biases` use bf16 in the encoder and float32 in the prediction/joint networks.

## Files

Sizes in bytes and SHA-256 values were checked against `var/dist/publish/mlx-8bit/` with `ls -l` and `shasum -a 256`.

| File | Size | SHA-256 |
|---|---:|---|
| `model.safetensors` | 727,881,324 | `17ec1d12a67b35303c1bbc4defa3077b03f14665550e45d2749ada12ef73368e` |
| `config.json` | 726 | `0b7ad196af39a5120570d31bdcb0c9194e89d216892c9ea8e4d552170e8df6b3` |
| `tokenizer.model` | 279,711 | `9bd7bcdff62e43200001cd4c258d22e475b7b3b1e0f64f8be8e6e046f402f38c` |
| `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `NOTICE` | 741 | `cb6145131e3cb48e0bb432aa5b949ae72330ef0dddaf1621de9b0cc65ab8b51b` |

## Limitations

The [bf16 limitations](https://huggingface.co/basyauma-tech/Haas-ja-v1.0-mlx#limitations) also apply. Quantized inference is unmeasured on long audio with splitting, silence/noise, JSUT, TEDxJP-10K, SpeechBSD, group sizes other than 64, low-memory machines and batch sizes ≥2.

## License and attribution

- Same as bf16: Apache-2.0 for weights/code. Source weights: [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0), Copyright Third Intelligence, Inc. Attribution: `NOTICE`. "Haas" identifies the source.
- References: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo), [third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR) (official inference package), [matsuurajun/haas-mlx](https://github.com/matsuurajun/haas-mlx), [Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx): Apache-2.0. [flash-linear-attention](https://github.com/fla-org/flash-linear-attention): MIT.
- Sanami's float32 ONNX is used only for numerical comparison; its weights are not redistributed.
- Evaluation data: FLEURS CC BY 4.0; JVNV CC BY-SA 4.0. Neither included nor redistributed.
