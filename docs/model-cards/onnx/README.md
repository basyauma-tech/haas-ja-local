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

# Haas-ja-v1.0-onnx

[English version below](#english)

**非公式の変換物です。** [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)のONNX版です。Third Intelligence, Inc.とは提携・承認・後援の関係にありません。

fp32は元のbf16をfloat32化。int8はONNX Runtimeの動的量子化版です。Hugging Face上の分類はquantizedですが、量子化はint8のみ。**既定はfp32、int8は参考扱い**です。int8はFLEURSの精度基準を満たしませんでした。

## 概要・環境

- ONNX Runtime対応CPU向け。NeMo・CUDA・Triton不要。推論コード: [haas-ja-local](https://github.com/basyauma-tech/haas-ja-local)。
- Python >=3.12,<3.13。実行時依存はonnxruntime・numpy・sentencepiece・soundfile・soxr。torch不要。
- 計測はApple M5 Max（128GB）のCPUのみ。Superコア6・Performanceコア12、macOS 27.2（26B123）、Python 3.12.15、onnxruntime 1.31.1（`CPUExecutionProvider`、`ORT_ENABLE_ALL`）、numpy 2.5.4。
- x86_64での速度・精度、Linux・Windowsは未計測。int8の精度は命令セットで変わる可能性あり。

## 使い方

`hf`は`huggingface_hub`に付属（必要なら`pip install huggingface_hub`）。

```sh
pip install "haas-ja-local[onnx] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download basyauma-tech/Haas-ja-v1.0-onnx --local-dir Haas-ja-v1.0-onnx
```

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("Haas-ja-v1.0-onnx")
result = transcribe(backend, tokenizer, load_audio("audio.wav"))
print(result.text)
backend, tokenizer = load_model("Haas-ja-v1.0-onnx", variant="int8", threads=4)
```

```sh
haas-ja-local transcribe Haas-ja-v1.0-onnx audio.wav --backend onnx --variant fp32 --threads 4
haas-ja-local transcribe Haas-ja-v1.0-onnx audio.wav --backend onnx --variant int8 --threads 4
```

- `fp32/`があれば既定はfp32、それ以外はint8。
- 入力はsoundfile対応形式（WAV・FLACなど）。複数チャンネルを平均してモノラル化し、soxrで16kHzへ変換。
- 30秒超は逐次分割。各断片の先頭から30秒±2秒内で、最も静かな0.1秒窓の中心を切断点とし、認識結果を連結。
- `--json --timestamps`で1ファイル1行のJSONとトークン開始・終了時刻を出力。sherpa-onnxでの動作は未確認。

## 精度

[asr-ja-bench](https://huggingface.co/datasets/third-intelligence/asr-ja-bench) v1.1、[jacer](https://github.com/third-intelligence/jacer) v1.2.0。単位は%。FLEURSは650発話（2.36時間）、JVNVは1,375発話（3.33時間）。

| 版 | サブセット | CER strict | CER flex | CER kana | 公表値strict | 差（ポイント） | 判定 |
|---|---|---:|---:|---:|---:|---:|---|
| fp32 | FLEURS | 3.69 | 3.02 | 2.38 | 3.65〜3.67 | +0.02 | 合格 |
| fp32 | JVNV | 4.52 | 3.06 | 2.25 | 4.50〜4.53 | 0.00 | 合格 |
| int8 | FLEURS | 4.05 | 3.35 | 2.50 | 3.65〜3.67 | +0.38 | 不合格 |
| int8 | JVNV | 4.43 | 3.16 | 2.33 | 4.50〜4.53 | -0.07 | 合格 |

- 上記M5 MaxのCPU、バッチ1、8スレッド。発話ごとに認識し、サブセット全体でCERを計算（マイクロ平均）。句読点・空白を除外。全発話30秒以下で分割なし。
- データは`rebuild.py`で再構築し、書き起こしのハッシュを照合済み。JVNVの16kHz化は想定のffmpeg 7.1ではなく9.0.2のため、公式音声と微差の可能性あり。
- 公表値は公式のNeMo・bf16・H200での測定。モデルカードと公開記事の差を幅で表示。差は近い端との差、範囲内は0。合格基準は公表値との差0.3ポイント以内。flex・kanaの公表値は未照合で、差・合否は判定していない。
- 精度調整なし。int8でデコーダをfp32にしてもFLEURSのCER strictは4.09%で改善せず。配布int8はデコーダも量子化。
- 未測定: JSUT・TEDxJP-10K・SpeechBSD（全サブセットの重み付きCERなし）。

詳細・再現手順: [ONNXの検証記録](https://github.com/basyauma-tech/haas-ja-local/blob/main/docs/verification/2026-11-08-mac-onnx.md)、[PyTorch参照実装の検証](https://github.com/basyauma-tech/haas-ja-local/blob/main/docs/verification/2026-11-08-mac-numerics.md)。

## 実装の確認

fp32とPyTorch参照実装（float32）はFLEURS全650発話で本文が一致。6発話の数値比較では最大誤差がlog-mel 9.54e-07、エンコーダ出力（相対）3.82e-06、結合ネットワーク出力（相対）1.11e-06で、トークン列はすべて一致しました。

PyTorch参照実装自体も第三者のfloat32 ONNXと6発話で比較済み。最大誤差はlog-mel 9.54e-07、エンコーダ（相対）4.36e-06、結合ネットワーク（相対）1.16e-06で、トークン列はすべて一致しました。

## 速度とメモリ

上記M5 MaxのCPU、バッチ1で逐次認識。FLEURSの先頭3発話をウォームアップとして除いた647発話、2.35時間。モデル・音声読み込みを除く認識時間で集計。GB＝10^9バイト。

| 版 | スレッド数 | RTFx中央値 | RTFx合計 | ピークRSS |
|---|---:|---:|---:|---:|
| fp32 | 4 | 72.4 | 72.7 | 3.24GB |
| fp32 | 12 | 66.7 | 66.4 | 3.24GB |
| int8 | 4 | 70.0 | 69.8 | 1.41GB |
| int8 | 12 | 84.3 | 84.3 | 1.40GB |

- RTFx合計は合計音声÷合計処理時間。ピークRSSは`getrusage`の`ru_maxrss`。
- 12スレッドは全コアではなく、Super・Performanceコアにまたがる。fp32は4スレッドのほうが速く、再実行でも同じ傾向。原因は未調査。
- CLIのfp32で33.8分（2,026.5秒）を68分割: 起動・モデル読み込み込み23.30秒（RTFx 87.0）、最大RSS 3.52GB。スレッド数は既定（ONNX Runtimeの自動設定）。
- 公表RTFxはH200・バッチ512のため比較不可。4・12以外のスレッド数での速度は未計測。

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
| SHA-256 | `c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa`（`config.json`の`source_sha256`） |
| サイズ | 1,232,384,000バイト |
| 変換ツール | haas-ja-local 0.1.0、書き出し`e4968f2`、評価・計測`3d1050b` |
| 書き出し環境 | torch 2.15.0、onnx 1.23.2、onnxruntime 1.31.1 |

- PyTorch参照実装からONNXへ書き出し。
- int8はper-channel動的量子化、`reduce_range=False`、量子化前の前処理なし。エンコーダ・デコーダとも量子化し、エンコーダ重みは両版とも外部`.data`に保存。
- `--exclude-decoder`で再書き出しした際、fp32の全ファイルとint8のエンコーダはバイト単位で一致。別の量子化設定は未測定。

## ファイル一覧

合計サイズ: fp32約2.46GB、int8約0.62GB、リポジトリ全体約3.1GB。

サイズはバイト。SHA-256は各版の`SHA256SUMS`から転記。「—」は同ファイルに記載なし。

| ファイル | サイズ | SHA-256 |
|---|---:|---|
| `fp32/encoder.onnx` | 740,878 | `a93c1ac09be3a82cfd7d357affe4c84c2f4746a9f831f1ad04374fb744107adb` |
| `fp32/encoder.onnx.data` | 2,435,420,160 | `88689ebfe6a760d373598298b56716ca74ffee7f21e23d6a47a28a2557ad20f1` |
| `fp32/decoder_joint.onnx` | 27,712,720 | `9e1e99ec8d4a0ea43b19f8572b3fd02beae2dfb7f671e2ff975dd0d7428d626a` |
| `int8/encoder.onnx` | 3,198,708 | `f36c5a24e5a43c58321244b9b396f15f00277a48e93574a2bb79308c17cd3234` |
| `int8/encoder.onnx.data` | 609,822,464 | `03db20b51fd813f6b694309520621f66ea331dc7769f7679d936237a0abdb27a` |
| `int8/decoder_joint.onnx` | 7,137,058 | `ba4726a5a72f2673407435750f893d40c8cda04a34ac0a4a71c389802cf62c73` |
| `fp32/SHA256SUMS` | 248 | — |
| `int8/SHA256SUMS` | 248 | — |
| `config.json` | 640 | — |
| `tokenizer.model` | 279,711 | — |
| `preprocessor.npz` | 133,686 | — |
| `LICENSE` | 11,358 | — |
| `NOTICE` | 741 | — |

共通ファイルはモデル設定、SentencePieceトークナイザ、前処理の窓・メルフィルタバンク。両版のチェックサムは検証済み。

### 入出力仕様

opset 17。Bはバッチ、Tは入力フレーム数、T'はエンコード後のフレーム数。

| ネットワーク | 入出力 | 名前・形状・型 |
|---|---|---|
| エンコーダ | 入力 | `audio_signal [B,128,T] float32`、`length [B] int64` |
| エンコーダ | 出力 | `outputs [B,1024,T'] float32`、`encoded_lengths [B] int64` |
| デコーダ＋結合 | 入力 | `encoder_outputs [B,1024,1] float32`、`targets [B,1] int32`、`target_length [B] int32`、`input_states_1/2 [2,B,20480] float32` |
| デコーダ＋結合 | 出力 | `outputs [B,1,1,3078] float32`（語彙3072＋blank＋duration 5）、`prednet_lengths [B] int32`、`output_states_1/2 [2,B,20480] float32` |

`input_states_1`はKDAの再帰状態5×64×64。`input_states_2`はq/k/vのshort convバッファ3×320×4を先頭に置き、残りは0。

## 制限事項

- 元モデル同様、フィラーが抜けやすく、漢字・かな・数字の表記指定は不可。無音・雑音で短語を出す可能性あり。
- 30秒超の分割でCERが悪化。FLEURS 150発話を連結した33.8分音声のfp32のCER strictは、発話別4.10%、分割後6.39%。分割後のstrict/flex/kanaは6.39/5.70/4.62%でPyTorchと一致し、本文も同一。ONNXの数値誤差による悪化ではない。
- 同一話者が話し続ける実際の長時間録音は未測定。
- バッチ1のみ。ストリーミング・話者ラベル・公式Haas-ASRのカスタム語彙は未対応。
- 時刻は80ms刻みのトークン単位。単語単位ではない。

## ライセンスと帰属

- 重み・推論コード: Apache-2.0。元重みは[third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)（Copyright Third Intelligence, Inc.）。帰属は`NOTICE`。「Haas」は変換元を示す名称。
- 参考実装: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)、[third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR)（公式推論パッケージ）、[matsuurajun/haas-mlx](https://github.com/matsuurajun/haas-mlx)、[Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx)はApache-2.0。[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)はMIT。
- 他者の変換物は再配布しません。SanamiのONNXは数値比較のみ。
- 評価データ: FLEURSはCC BY 4.0、JVNVはCC BY-SA 4.0。同梱・再配布しません。

---

# English

**This is an unofficial conversion** of [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) to ONNX. It is not affiliated with, endorsed by, or sponsored by Third Intelligence, Inc.

fp32 casts the original bf16 to float32; int8 uses ONNX Runtime dynamic quantization. Hugging Face classifies this as quantized, but only int8 is quantized. **fp32 is the default; int8 is for reference.** int8 failed the FLEURS accuracy threshold.

## Overview and requirements

- CPU inference with ONNX Runtime; no NeMo, CUDA or Triton. Inference code: [haas-ja-local](https://github.com/basyauma-tech/haas-ja-local).
- Python >=3.12,<3.13; runtime dependencies: onnxruntime, numpy, sentencepiece, soundfile, soxr. No torch needed.
- Measured only on the CPU of an Apple M5 Max (128 GB): 6 Super and 12 Performance cores, macOS 27.2 (26B123), Python 3.12.15, onnxruntime 1.31.1 (`CPUExecutionProvider`, `ORT_ENABLE_ALL`), numpy 2.5.4.
- Speed and accuracy on x86_64, Linux and Windows are unmeasured. int8 accuracy may depend on the instruction set.

## Usage

`hf` comes with `huggingface_hub` (`pip install huggingface_hub` if needed).

```sh
pip install "haas-ja-local[onnx] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download basyauma-tech/Haas-ja-v1.0-onnx --local-dir Haas-ja-v1.0-onnx
```

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("Haas-ja-v1.0-onnx")
result = transcribe(backend, tokenizer, load_audio("audio.wav"))
print(result.text)
backend, tokenizer = load_model("Haas-ja-v1.0-onnx", variant="int8", threads=4)
```

```sh
haas-ja-local transcribe Haas-ja-v1.0-onnx audio.wav --backend onnx --variant fp32 --threads 4
haas-ja-local transcribe Haas-ja-v1.0-onnx audio.wav --backend onnx --variant int8 --threads 4
```

- Defaults to fp32 when `fp32/` exists, otherwise int8.
- Input: soundfile formats (WAV, FLAC, etc.). Channels are averaged to mono; soxr resamples to 16 kHz.
- Audio above 30 seconds is split sequentially at the centre of the quietest 0.1-second window within 30 ± 2 seconds of each piece's start. Results are joined.
- `--json --timestamps` emits one JSON line per file with token start/end times. Execution with sherpa-onnx is unverified.

## Accuracy

[asr-ja-bench](https://huggingface.co/datasets/third-intelligence/asr-ja-bench) v1.1, [jacer](https://github.com/third-intelligence/jacer) v1.2.0; values in %. FLEURS: 650 utterances (2.36 hours); JVNV: 1,375 utterances (3.33 hours).

| Variant | Subset | CER strict | CER flex | CER kana | Published strict | Difference (points) | Result |
|---|---|---:|---:|---:|---:|---:|---|
| fp32 | FLEURS | 3.69 | 3.02 | 2.38 | 3.65–3.67 | +0.02 | Pass |
| fp32 | JVNV | 4.52 | 3.06 | 2.25 | 4.50–4.53 | 0.00 | Pass |
| int8 | FLEURS | 4.05 | 3.35 | 2.50 | 3.65–3.67 | +0.38 | Fail |
| int8 | JVNV | 4.43 | 3.16 | 2.33 | 4.50–4.53 | -0.07 | Pass |

- M5 Max CPU setup above, batch 1, 8 threads. Utterances transcribed individually; CER computed across each subset (micro-average), ignoring punctuation/spaces. All utterances ≤30 seconds; no splitting.
- Data rebuilt with `rebuild.py`; transcript hashes verified. JVNV resampling to 16 kHz used ffmpeg 9.0.2 instead of the expected 7.1, so audio may differ slightly.
- Published values use official NeMo, bf16 and H200. Model-card/blog differences form the range. Differences use the nearer endpoint, or 0 within the range. Acceptance threshold: within 0.3 points of published values. Published flex/kana values were not compared; no differences or pass/fail results are assigned to them.
- No accuracy tuning. Keeping the int8 decoder in fp32 gave FLEURS CER strict 4.09%, with no improvement. The distributed int8 variant also quantizes the decoder.
- Unmeasured: JSUT, TEDxJP-10K and SpeechBSD (no weighted CER across all subsets).

Details/reproduction (Japanese): [ONNX verification](https://github.com/basyauma-tech/haas-ja-local/blob/main/docs/verification/2026-11-08-mac-onnx.md), [PyTorch reference verification](https://github.com/basyauma-tech/haas-ja-local/blob/main/docs/verification/2026-11-08-mac-numerics.md).

## Implementation verification

fp32 and the PyTorch reference (float32) produced identical text for all 650 FLEURS utterances. On 6 utterances, maximum errors were 9.54e-07 for log-mel, 3.82e-06 for encoder output (relative), and 1.11e-06 for joint output (relative); all token sequences matched.

The PyTorch reference itself was also compared against third-party float32 ONNX on 6 utterances: maximum errors were 9.54e-07 for log-mel, 4.36e-06 for encoder output (relative), and 1.16e-06 for joint output (relative); all token sequences matched.

## Speed and memory

M5 Max CPU setup above, batch 1, sequential recognition. 647 FLEURS utterances, 2.35 hours, excluding the first 3 warm-ups. Recognition timing excludes model/audio loading. GB = 10^9 bytes.

| Variant | Threads | Median RTFx | Aggregate RTFx | Peak RSS |
|---|---:|---:|---:|---:|
| fp32 | 4 | 72.4 | 72.7 | 3.24 GB |
| fp32 | 12 | 66.7 | 66.4 | 3.24 GB |
| int8 | 4 | 70.0 | 69.8 | 1.41 GB |
| int8 | 12 | 84.3 | 84.3 | 1.40 GB |

- Aggregate RTFx = total audio / total processing time. Peak RSS uses `getrusage`'s `ru_maxrss`.
- 12 threads span Super and Performance cores, not all cores. fp32 was faster with 4 threads, including on rerun; the cause is uninvestigated.
- CLI fp32, 33.8 minutes (2,026.5 seconds), 68 pieces: 23.30 seconds including startup/model loading (RTFx 87.0), maximum RSS 3.52 GB; default thread count (ONNX Runtime automatic configuration).
- Published RTFx uses H200, batch 512; not comparable. Speed with thread counts other than 4 and 12 is unmeasured.

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
| SHA-256 | `c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa` (`source_sha256` in `config.json`) |
| Size | 1,232,384,000 bytes |
| Converter | haas-ja-local 0.1.0; export `e4968f2`, evaluation/measurement `3d1050b` |
| Export environment | torch 2.15.0, onnx 1.23.2, onnxruntime 1.31.1 |

- Exported from the PyTorch reference to ONNX.
- int8: per-channel dynamic quantization, `reduce_range=False`, no pre-quantization preprocessing. Both encoder and decoder are quantized; both variants store encoder weights in external `.data` files.
- Re-exporting with `--exclude-decoder` gave byte-identical fp32 files and int8 encoder files. Other quantization settings are unmeasured.

## Files

Total size: fp32 approximately 2.46 GB, int8 approximately 0.62 GB, entire repository approximately 3.1 GB.

Sizes in bytes. SHA-256 values are copied from each variant's `SHA256SUMS`; “—” means not listed there.

| File | Size | SHA-256 |
|---|---:|---|
| `fp32/encoder.onnx` | 740,878 | `a93c1ac09be3a82cfd7d357affe4c84c2f4746a9f831f1ad04374fb744107adb` |
| `fp32/encoder.onnx.data` | 2,435,420,160 | `88689ebfe6a760d373598298b56716ca74ffee7f21e23d6a47a28a2557ad20f1` |
| `fp32/decoder_joint.onnx` | 27,712,720 | `9e1e99ec8d4a0ea43b19f8572b3fd02beae2dfb7f671e2ff975dd0d7428d626a` |
| `int8/encoder.onnx` | 3,198,708 | `f36c5a24e5a43c58321244b9b396f15f00277a48e93574a2bb79308c17cd3234` |
| `int8/encoder.onnx.data` | 609,822,464 | `03db20b51fd813f6b694309520621f66ea331dc7769f7679d936237a0abdb27a` |
| `int8/decoder_joint.onnx` | 7,137,058 | `ba4726a5a72f2673407435750f893d40c8cda04a34ac0a4a71c389802cf62c73` |
| `fp32/SHA256SUMS` | 248 | — |
| `int8/SHA256SUMS` | 248 | — |
| `config.json` | 640 | — |
| `tokenizer.model` | 279,711 | — |
| `preprocessor.npz` | 133,686 | — |
| `LICENSE` | 11,358 | — |
| `NOTICE` | 741 | — |

Shared files provide model configuration, the SentencePiece tokenizer and preprocessing window/mel filterbank. Both checksum manifests have been verified.

### Input/output specification

Opset 17. B is batch size, T is input frames, and T' is encoded frames.

| Network | Direction | Names, shapes and types |
|---|---|---|
| Encoder | Input | `audio_signal [B,128,T] float32`, `length [B] int64` |
| Encoder | Output | `outputs [B,1024,T'] float32`, `encoded_lengths [B] int64` |
| Decoder + joint | Input | `encoder_outputs [B,1024,1] float32`, `targets [B,1] int32`, `target_length [B] int32`, `input_states_1/2 [2,B,20480] float32` |
| Decoder + joint | Output | `outputs [B,1,1,3078] float32` (vocabulary 3072 + blank + duration 5), `prednet_lengths [B] int32`, `output_states_1/2 [2,B,20480] float32` |

`input_states_1` holds the KDA recurrent state, 5×64×64. `input_states_2` holds q/k/v short conv buffers, 3×320×4, at the beginning; the remainder is 0.

## Limitations

- As with the original model, fillers may be omitted; kanji/kana/numeral style cannot be selected. Silence/noise may produce short words.
- Splitting above 30 seconds raises CER. For 150 FLEURS utterances joined into 33.8 minutes, fp32 CER strict was 4.10% individually and 6.39% after splitting. Split strict/flex/kana scores were 6.39/5.70/4.62%, matching PyTorch; text was also identical. The degradation is not due to ONNX numerical error.
- Real long recordings of one continuous speaker are unmeasured.
- Batch 1 only. No streaming, speaker labels or official Haas-ASR custom vocabulary.
- Token timestamps use 80 ms steps; they are not word timestamps.

## License and attribution

- Weights/code: Apache-2.0. Source weights: [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0), Copyright Third Intelligence, Inc. Attribution: `NOTICE`. “Haas” identifies the source.
- References: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo), [third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR) (official inference package), [matsuurajun/haas-mlx](https://github.com/matsuurajun/haas-mlx), [Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx): Apache-2.0. [flash-linear-attention](https://github.com/fla-org/flash-linear-attention): MIT.
- No third-party conversions are redistributed. Sanami's ONNX is used only for numerical comparison.
- Evaluation data: FLEURS is CC BY 4.0; JVNV is CC BY-SA 4.0. Neither is included or redistributed.
