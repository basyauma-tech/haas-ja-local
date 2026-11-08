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

**非公式の変換物です。** [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)のONNX版です。Third Intelligence, Inc.とは提携・承認・後援の関係にありません。

fp32は元のbf16をfloat32化。int8はONNX Runtimeで動的量子化した8ビット整数版。Hugging Face分類はquantizedですが、量子化はint8のみ。

## 概要・環境

- ONNX Runtime対応CPU（macOS・Linux・Windows）。NeMo・CUDA・Triton不要。推論コード: [haas-ja-local](https://github.com/basyauma-tech/haas-ja-local)（別途インストール）。
- Python >=3.12,<3.13。onnxruntime・numpy・sentencepiece・soundfile・soxrに依存。torch不要。
- TODO(Task 12/13): 検証機種・CPU・OS・onnxruntimeの版、未検証環境。
- TODO(Task 12/13): `fp32/encoder.onnx`・`fp32/encoder.onnx.data`の実サイズ。
- TODO(Task 12/13): int8のper-channel動的量子化設定（`reduce_range`・デコーダのfp32維持）、`.data`の有無・サイズ。

TODO(Task 12/13): fp32/int8の既定と理由。

## 使い方

`hf`は`huggingface_hub`に付属。

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

CLI（`--threads`はONNX Runtimeのスレッド数）:

```sh
haas-ja-local transcribe Haas-ja-v1.0-onnx audio.wav --threads 4
```

- `fp32/`があればfp32、それ以外はint8。int8は`--variant int8`か`HaasONNX.load(path, variant="int8")`で指定。TODO(Task 12/13): int8のCER測定後に既定を見直す。
- 長時間音声・時刻はMLX版と同じ。30秒超を約30秒ごとに分割。`--json --timestamps`でトークン時刻を出力。

TODO(Task 12/13): sherpa-onnxの動作確認（動かなければ「未確認」）。

## 精度

TODO(Task 12/13): asr-ja-bench v1.1のFLEURS・JVNVでfp32/int8のCER strict/flex/kana、公表値、差を表にする。条件: 機種・CPU・OS・onnxruntime版・スレッド数・jacer v1.2.0。未測定一覧とtorch（float32）との数値一致も記載。

## 速度とメモリ

TODO(Task 12/13): fp32/int8、4スレッド/全コアのRTFx・ピークメモリ表。条件: バッチ1・逐次認識・CPU名・OS。公表RTFx（H200・バッチ512）とは比較不可と明記。

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

TODO(Task 12/13): float32化・トレース方式・int8量子化設定。

## ファイル一覧

| ファイル | 内容 |
|---|---|
| `fp32/encoder.onnx`、`fp32/encoder.onnx.data` | エンコーダのグラフ・外部重み |
| `fp32/decoder_joint.onnx` | デコーダ＋結合ネットワーク |
| `int8/encoder.onnx`、`int8/encoder.onnx.data`、`int8/decoder_joint.onnx` | int8版（`.data`の有無は上記TODOで確認） |
| `config.json`、`tokenizer.model` | モデル設定・トークナイザ |
| `preprocessor.npz` | 前処理用の窓・メルフィルタバンク |
| `SHA256SUMS` | ファイルのSHA-256一覧 |
| `LICENSE`、`NOTICE` | ライセンス・帰属情報 |

## 入出力仕様

opset 17。Bはバッチ、Tは入力フレーム数、T'はエンコード後のフレーム数。

| ネットワーク | 入出力 | 名前・形状・型 |
|---|---|---|
| エンコーダ | 入力 | `audio_signal [B,128,T] float32`、`length [B] int64` |
| エンコーダ | 出力 | `outputs [B,1024,T'] float32`、`encoded_lengths [B] int64` |
| デコーダ＋結合 | 入力 | `encoder_outputs [B,1024,1] float32`、`targets [B,1] int32`、`target_length [B] int32`、`input_states_1/2 [2,B,20480] float32` |
| デコーダ＋結合 | 出力 | `outputs [B,1,1,3078] float32`（語彙3072＋blank＋duration 5）、`prednet_lengths [B] int32`、`output_states_1/2 [2,B,20480] float32` |

`input_states_1`はKDAの再帰状態5×64×64。`input_states_2`はq/k/vのshort convバッファ3×320×4を先頭に置き、残りは0。

## 制限事項

- 元モデル同様、フィラーが抜けやすく、漢字・かな・数字の表記指定は不可。
- 無音・雑音で短語を出す可能性あり。
- 30秒超の分割でCERが悪化。TODO(Task 12/13): ONNXの数値かMLX記録への参照。
- バッチ1のみ。ストリーミング・話者ラベル・公式Haas-ASRのカスタム語彙は未対応。
- 時刻は80ms刻みのトークン単位。単語単位ではない。

## ライセンスと帰属

- 重み・推論コード: Apache-2.0。元重みは[third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)（Copyright Third Intelligence, Inc.）。帰属は`NOTICE`。「Haas」は変換元を示す名称。
- 参考実装: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)、[third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR)（公式推論パッケージ）、[matsuurajun/haas-mlx](https://github.com/matsuurajun/haas-mlx)、[Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx)はApache-2.0。[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)はMIT。
- SanamiのONNXは数値比較のみで、重みは再配布しない。配布重みはすべて元の`.nemo`の非公式の変換物。
- [評価セット](https://huggingface.co/datasets/third-intelligence/asr-ja-bench)、[CER計算](https://github.com/third-intelligence/jacer)。FLEURSはCC BY 4.0、JVNVはCC BY-SA 4.0。評価データは同梱・再配布しない。

---

# English

**This is an unofficial conversion** of [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) to ONNX. It is not affiliated with, endorsed by, or sponsored by Third Intelligence, Inc.

fp32: original bf16 cast to float32. int8: ONNX Runtime dynamic 8-bit quantization. Hugging Face classification: quantized; only int8 is quantized.

## Overview and requirements

- ONNX Runtime CPU, macOS/Linux/Windows; no NeMo/CUDA/Triton. Install separate inference code: [haas-ja-local](https://github.com/basyauma-tech/haas-ja-local).
- Python >=3.12,<3.13; onnxruntime, numpy, sentencepiece, soundfile, soxr. No torch.
- TODO(Task 12/13): tested machine, CPU, OS, onnxruntime version; untested environments.
- TODO(Task 12/13): actual `fp32/encoder.onnx`/`fp32/encoder.onnx.data` size.
- TODO(Task 12/13): int8 per-channel dynamic quantization settings (`reduce_range`, fp32 decoder), `.data` presence and sizes.

TODO(Task 12/13): fp32/int8 default and rationale.

## Usage

`hf` comes with `huggingface_hub`.

```sh
pip install "haas-ja-local[onnx] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download basyauma-technology/Haas-ja-v1.0-onnx --local-dir Haas-ja-v1.0-onnx
haas-ja-local transcribe Haas-ja-v1.0-onnx audio.wav --threads 4
```

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("Haas-ja-v1.0-onnx")
print(transcribe(backend, tokenizer, load_audio("audio.wav")).text)
```

`--threads` sets ONNX Runtime threads.

- Uses fp32 if `fp32/` exists, otherwise int8. Select int8 with `--variant int8` or `HaasONNX.load(path, variant="int8")`. TODO(Task 12/13): revisit the default after measuring int8 CER.
- Long audio/times match MLX: split audio above 30 seconds into ~30-second pieces; `--json --timestamps` emits token times.

TODO(Task 12/13): verify sherpa-onnx execution; otherwise mark "not checked".

## Accuracy

TODO(Task 12/13): asr-ja-bench v1.1 FLEURS/JVNV table: fp32/int8 CER strict/flex/kana, published values, differences. Include machine/CPU/OS/onnxruntime/threads/jacer v1.2.0, unmeasured cases, torch float32 agreement.

## Speed and memory

TODO(Task 12/13): fp32/int8 RTFx/peak memory, 4 threads/all cores; batch 1, sequential, CPU/OS. Published H200/batch 512 RTFx is not comparable.

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

TODO(Task 12/13): float32 cast, tracing method, int8 quantization settings.

## Files

| File | Contents |
|---|---|
| `fp32/encoder.onnx`, `fp32/encoder.onnx.data` | Encoder graph and external weights |
| `fp32/decoder_joint.onnx` | Decoder and joint network |
| `int8/encoder.onnx`, `int8/encoder.onnx.data`, `int8/decoder_joint.onnx` | int8 variant (`.data` presence pending the TODO above) |
| `config.json`, `tokenizer.model` | Model configuration and tokenizer |
| `preprocessor.npz` | Preprocessing window and mel filterbank |
| `SHA256SUMS` | File SHA-256 checksums |
| `LICENSE`, `NOTICE` | License and attribution |

## Input/output specification

Opset 17. B is batch size, T is the input frame count, and T' is the encoded frame count.

| Network | Direction | Names, shapes and types |
|---|---|---|
| Encoder | Input | `audio_signal [B,128,T] float32`, `length [B] int64` |
| Encoder | Output | `outputs [B,1024,T'] float32`, `encoded_lengths [B] int64` |
| Decoder + joint | Input | `encoder_outputs [B,1024,1] float32`, `targets [B,1] int32`, `target_length [B] int32`, `input_states_1/2 [2,B,20480] float32` |
| Decoder + joint | Output | `outputs [B,1,1,3078] float32` (vocabulary 3072 + blank + duration 5), `prednet_lengths [B] int32`, `output_states_1/2 [2,B,20480] float32` |

`input_states_1` holds the KDA recurrent state, 5×64×64. `input_states_2` holds the q/k/v short conv buffers, 3×320×4, at the beginning; the remainder is 0.

## Limitations

- As with the original model, fillers may be omitted; kanji/kana/numeral style cannot be selected.
- Silence/noise may produce short words.
- Splitting above 30 seconds raises CER. TODO(Task 12/13): ONNX numbers or reference to the MLX record.
- Batch 1 only. No streaming, speaker labels or official Haas-ASR custom vocabulary.
- Token timestamps in 80 ms steps, not word timestamps.

## License and attribution

- Weights/code: Apache-2.0. Source weights: [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0), Copyright Third Intelligence, Inc. Attribution: `NOTICE`. "Haas" identifies the source.
- References: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo), [third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR) (official inference package), [matsuurajun/haas-mlx](https://github.com/matsuurajun/haas-mlx), [Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx): Apache-2.0. [flash-linear-attention](https://github.com/fla-org/flash-linear-attention): MIT.
- Sanami's ONNX is used only for numerical comparison; its weights are not redistributed. All distributed weights are unofficial conversions of the original `.nemo`.
- [Benchmark](https://huggingface.co/datasets/third-intelligence/asr-ja-bench), [CER library](https://github.com/third-intelligence/jacer). FLEURS: CC BY 4.0; JVNV: CC BY-SA 4.0. Evaluation data is neither included nor redistributed.
