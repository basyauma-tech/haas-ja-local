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

**非公式の変換物です。** [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)のMLX版です。Third Intelligence, Inc.とは提携・承認・後援の関係にありません。

重みは元のbf16の値を保持し、軸の並べ替えと推論に不要な`num_batches_tracked`の除去だけを行います。Hugging Face上の分類はquantizedですが、量子化していません。

線形層を8bitに量子化した版（約0.73GB）: [basyauma-tech/Haas-ja-v1.0-mlx-8bit](https://huggingface.co/basyauma-tech/Haas-ja-v1.0-mlx-8bit)

## 概要・環境

元モデルはFastConformerエンコーダとTDTデコーダ（予測ネットワークはKimi Delta Attention）で、公式の実行環境はLinuxとNVIDIA GPU（NeMo 2.7）に限られます。

- この変換物: Apple SiliconのMac専用。NeMo・CUDA・Tritonは不要。推論コードは別途[haas-ja-local](https://github.com/basyauma-tech/haas-ja-local)を使用。
- Python 3.12（>=3.12,<3.13）。実行時依存はmlx・numpy・sentencepiece・soundfile・soxr。torch不要。重み用に約1.2GB必要。
- 検証機はApple M5 Max（128GB）、macOS 27.2、mlx 0.32.4の1台のみ。

## 使い方

`hf`は`huggingface_hub`に付属（必要なら`pip install huggingface_hub`）。

```sh
pip install "haas-ja-local[mlx] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download basyauma-tech/Haas-ja-v1.0-mlx --local-dir Haas-ja-v1.0-mlx
```

Python:

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("Haas-ja-v1.0-mlx")
result = transcribe(backend, tokenizer, load_audio("audio.wav"))
print(result.text)
```

CLI:

```sh
haas-ja-local transcribe Haas-ja-v1.0-mlx audio.wav
```

- 入力: soundfile対応形式（WAV・FLACなど）。複数チャンネルは平均してモノラル化し、soxrで16kHzに変換。
- 長時間音声: 30秒超を逐次分割。各断片の先頭から30秒±2秒内で、最も静かな0.1秒窓の中心を切断点とし、認識結果を連結。
- Metalキャッシュ: 既定1GiB。`HaasMLX.load`の`cache_limit_bytes`で変更（`None`は設定しない）。
- 時刻: `--json --timestamps`で1ファイル1行のJSON。トークンごとの開始・終了秒を80ms刻みで出力。複数バイトトークンの文字は1件に統合。

```sh
haas-ja-local transcribe Haas-ja-v1.0-mlx audio.wav --json --timestamps
```

```json
{"file": "audio.wav", "text": "今日は良い天気ですね。", "tokens": [{"text": "今", "start": 0.0, "end": 0.32}, {"text": "日は", "start": 0.32, "end": 0.64}, {"text": "良", "start": 0.64, "end": 0.88}, {"text": "い", "start": 0.88, "end": 1.04}, {"text": "天", "start": 1.04, "end": 1.28}, {"text": "気", "start": 1.28, "end": 1.6}, {"text": "ですね", "start": 1.6, "end": 1.84}, {"text": "。", "start": 1.84, "end": 2.0}]}
```

## 精度

[asr-ja-bench](https://huggingface.co/datasets/third-intelligence/asr-ja-bench) v1.1、[jacer](https://github.com/third-intelligence/jacer) v1.2.0で測定。単位は%。

| サブセット | 発話数 | 音声（時間） | 指標 | この変換 | 公表値 | 差（ポイント） |
|---|---:|---:|---|---:|---:|---:|
| FLEURS | 650 | 2.36 | CER strict | 3.65 | 3.65〜3.67 | 0.00 |
| FLEURS | 650 | 2.36 | CER flex | 2.98 | 未照合 | - |
| FLEURS | 650 | 2.36 | CER kana | 2.33 | 未照合 | - |
| JVNV | 1,375 | 3.33 | CER strict | 4.51 | 4.50〜4.53 | 0.00 |
| JVNV | 1,375 | 3.33 | CER flex | 3.06 | 未照合 | - |
| JVNV | 1,375 | 3.33 | CER kana | 2.25 | 未照合 | - |

測定条件:

- `rebuild.py`で評価データを再構築し、全書き起こしのハッシュ照合に合格。JVNVの16kHz化は想定のffmpeg 7.1ではなく9.0.2のため、公式音声と微差の可能性あり。
- 1発話ずつ認識後、サブセット全体で`jacer.cer`を1回計算（マイクロ平均）。3指標とも句読点・空白を除外。全発話30秒以下で分割なし。
- 上記M5 Max、Metal、バッチ1。エンコーダbf16、予測・結合ネットワークfloat32。
- 公表値は公式asr-ja-bench v1.1（NeMo・bf16・H200）。モデルカードと公開記事の小数第2位の差を幅で表示。差は近い端との差、範囲内は0。flex・kanaは未照合。
- 精度調整なしの初回実行。

未測定: JSUT basic5000・TEDxJP-10K・SpeechBSD（5サブセットの重み付きCERなし）、Intel Mac、メモリ16GB以下、バッチ2以上、同一話者が話し続ける実際の長時間録音。

詳細・再現手順: [精度と速度](https://github.com/basyauma-tech/haas-ja-local/blob/main/docs/verification/2026-11-08-mac-mlx.md)、[数値一致](https://github.com/basyauma-tech/haas-ja-local/blob/main/docs/verification/2026-11-08-mac-numerics.md)。

MLX版（bf16）とPyTorch参照実装（float32）の比較では、FLEURS650発話中632発話（97.2%）でトークン列と本文が一致し、異なる18発話のうち12発話は句読点だけの違いでした。

## 実装の確認

PyTorch参照実装は、同じチェックポイントから第三者が書き出したfloat32のONNXとFLEURSの6発話で比較し、最大誤差はlog-mel 9.54e-07、エンコーダ出力（相対）4.36e-06、結合ネットワーク出力（相対）1.16e-06、トークン列は6発話すべて一致でした。

## 速度とメモリ

FLEURS全650発話から先頭3発話のウォームアップを除外した647発話、2.35時間。上記環境・計算型、バッチ1で逐次認識。音声読み込みを除く認識時間。GB＝10^9バイト。

| 項目 | 既定（キャッシュ上限1GiB） | 上限なし（参考、注） |
|---|---:|---:|
| RTFx（1発話ごとの中央値） | 254.2 | 250.9 |
| RTFx（合計音声÷合計処理時間） | 256.1 | 255.9 |
| ピークRSS | 1.57GB | 2.68GB |
| mlxのピーク割り当て（`mx.get_peak_memory()`） | 1.76GB | 1.82GB |
| macOSのピークメモリフットプリント | 3.19GB | 14.40GB |

- 上限なしのフットプリントのみ同条件の再実行値。他は初回。
- 上限なし: 1発話の中央値0.050秒、モデル読み込み0.15秒（OSキャッシュ済み）。
- CLIで33.8分（2,026.5秒）を68分割: 起動・モデル読み込み込み6.97秒、最大RSS 1.74GB、ピークフットプリント3.48GB。
- 元モデルの公表RTFxはNVIDIA H200・バッチ512のため比較不可。
- 同じ33.8分音声で上限なしではフットプリント13.96GB。実使用は1.25GBで一定。解放済みMetalバッファの保持でありリークではなく、既定の上限はこの増加を抑えるため。

## 変換方法

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

- bf16の値・型は不変。畳み込みのみMLXのチャンネル末尾配置へ変更: 2次元[O,I,H,W]→[O,H,W,I]、1次元（深さ方向を含む）[O,I,K]→[O,K,I]。
- 再変換のテンソル値はビット一致。safetensorsのメタデータキー順は不定のため、ファイル全体のSHA-256は一致しない場合あり。

## ファイル一覧

| ファイル | 内容 |
|---|---|
| `model.safetensors` | bf16、1,231,827,296バイト、SHA-256: `96600ed4ca8a4d9d45e6ed6bfd276ab9109da3f1c9e20e2bb0870e6f894fe5e4` |
| `config.json` | モデル設定 |
| `tokenizer.model` | SentencePiece BPE、3,072トークン |
| `LICENSE` | ライセンス |
| `NOTICE` | 帰属情報 |

## 制限事項

- 元モデル同様、フィラー（「えー」「あの」）が抜けやすく、漢字・かな・数字の表記指定は不可。
- 無音・雑音で短語を出す可能性あり。10秒の無音・弱いホワイトノイズの合成音声4種では出力なし。実録音は未確認。
- 30秒超の分割でCERが悪化。FLEURS 150発話を連結した33.8分音声のCER strictは、発話別4.10%、約30秒分割6.44%、PyTorch（float32）6.39%。MLXの数値誤差ではなく、モデルと公式`transcribe_longform`に沿った分割規則による。
- 悪化の約1ポイントは異話者の連結（発話を切らず30秒以内にまとめても5.08%）。同一話者の実録音では小さい可能性あり。残り約1.4ポイントは発話途中の切断（67境界中50）。
- バッチ1のみ。ストリーミング・話者ラベル・公式Haas-ASRのカスタム語彙は未対応。
- 時刻は80ms刻みのトークン単位。単語単位ではない。

## ライセンスと帰属

- 重み・推論コード: Apache-2.0。元重みは[third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)（Copyright Third Intelligence, Inc.）。帰属は`NOTICE`。「Haas」は変換元を示す名称。
- 参考実装: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)、[third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR)（公式推論パッケージ）、[matsuurajun/haas-mlx](https://github.com/matsuurajun/haas-mlx)、[Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx)はApache-2.0。[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)はMIT。
- Sanamiのfloat32 ONNXは数値比較のみで、重みは再配布しない。配布重みはすべて元の`.nemo`の非公式の変換物。
- 評価データ: FLEURSはCC BY 4.0、JVNVはCC BY-SA 4.0。同梱・再配布しない。

---

# English

**This is an unofficial conversion** of [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) to MLX. It is not affiliated with, endorsed by, or sponsored by Third Intelligence, Inc.

Original bf16 values are preserved; only tensor axes are rearranged and unused `num_batches_tracked` values removed. Hugging Face classifies it as quantized, but no quantization is applied.

Version with 8-bit linear layers (about 0.73 GB): [basyauma-tech/Haas-ja-v1.0-mlx-8bit](https://huggingface.co/basyauma-tech/Haas-ja-v1.0-mlx-8bit)

## Overview and requirements

The original model uses a FastConformer encoder and a TDT decoder (Kimi Delta Attention prediction network); the official runtime is limited to Linux with NVIDIA GPUs (NeMo 2.7).

- This conversion: Apple Silicon Macs only; no NeMo, CUDA or Triton. Inference code is separate: [haas-ja-local](https://github.com/basyauma-tech/haas-ja-local).
- Python 3.12 (>=3.12,<3.13); mlx, numpy, sentencepiece, soundfile, soxr. No torch needed. Allow about 1.2 GB for weights.
- Tested on one Apple M5 Max (128 GB), macOS 27.2, mlx 0.32.4.

## Usage

`hf` comes with `huggingface_hub` (`pip install huggingface_hub` if needed).

```sh
pip install "haas-ja-local[mlx] @ git+https://github.com/basyauma-tech/haas-ja-local"
hf download basyauma-tech/Haas-ja-v1.0-mlx --local-dir Haas-ja-v1.0-mlx
haas-ja-local transcribe Haas-ja-v1.0-mlx audio.wav
haas-ja-local transcribe Haas-ja-v1.0-mlx audio.wav --json --timestamps
```

```python
from haas_ja_local import load_audio, load_model, transcribe

backend, tokenizer = load_model("Haas-ja-v1.0-mlx")
print(transcribe(backend, tokenizer, load_audio("audio.wav")).text)
```

- Input: soundfile formats (WAV, FLAC, etc.). Channels are averaged to mono; soxr resamples to 16 kHz.
- Long audio: split sequentially above 30 seconds. Each cut is the centre of the quietest 0.1-second window within 30 ± 2 seconds of the piece's start. Results are joined.
- Metal cache: 1 GiB default; change `cache_limit_bytes` in `HaasMLX.load` (`None` leaves it unchanged).
- Times: `--json --timestamps` emits one JSON line per file, with token start/end seconds in 80 ms steps. Multiple byte tokens for one character are merged.

## Accuracy

[asr-ja-bench](https://huggingface.co/datasets/third-intelligence/asr-ja-bench) v1.1, [jacer](https://github.com/third-intelligence/jacer) v1.2.0; values in %.

| Subset | Utterances | Audio (hours) | Metric | This conversion | Published | Difference (points) |
|---|---:|---:|---|---:|---:|---:|
| FLEURS | 650 | 2.36 | CER strict | 3.65 | 3.65–3.67 | 0.00 |
| FLEURS | 650 | 2.36 | CER flex | 2.98 | not compared | - |
| FLEURS | 650 | 2.36 | CER kana | 2.33 | not compared | - |
| JVNV | 1,375 | 3.33 | CER strict | 4.51 | 4.50–4.53 | 0.00 |
| JVNV | 1,375 | 3.33 | CER flex | 3.06 | not compared | - |
| JVNV | 1,375 | 3.33 | CER kana | 2.25 | not compared | - |

Conditions:

- Data rebuilt with `rebuild.py`; all transcript hashes passed. JVNV resampling to 16 kHz used ffmpeg 9.0.2 instead of the expected 7.1, so audio may differ slightly.
- Utterances transcribed individually; one `jacer.cer` call per subset (micro-average). All three metrics ignore punctuation/spaces. All utterances ≤30 seconds; no splitting.
- M5 Max setup above, Metal, batch 1; bf16 encoder, float32 prediction/joint networks.
- Published: official asr-ja-bench v1.1 (NeMo, bf16, H200). Model-card/blog differences in the second decimal place form the range. Differences use the nearer endpoint, or 0 within the range. flex/kana not compared.
- First run; no accuracy tuning.

Not measured: JSUT basic5000, TEDxJP-10K, SpeechBSD (no weighted CER across all five subsets); Intel Macs; ≤16 GB machines; batch sizes ≥2; real long recordings of one continuous speaker.

Details/reproduction (Japanese): [accuracy and speed](https://github.com/basyauma-tech/haas-ja-local/blob/main/docs/verification/2026-11-08-mac-mlx.md), [numerical agreement](https://github.com/basyauma-tech/haas-ja-local/blob/main/docs/verification/2026-11-08-mac-numerics.md).

MLX (bf16) and the PyTorch reference (float32) matched in both token sequences and text on 632 of 650 FLEURS utterances (97.2%); 12 of the 18 differing utterances differed only in punctuation.

## Implementation verification

The PyTorch reference was compared on 6 FLEURS utterances against third-party float32 ONNX exported from the same checkpoint: maximum errors were 9.54e-07 for log-mel, 4.36e-06 for encoder output (relative), and 1.16e-06 for joint network output (relative); token sequences matched on all 6 utterances.

## Speed and memory

647 of 650 FLEURS utterances, 2.35 hours, excluding three warm-ups. Same setup/precision as above, batch 1, sequential. Transcription time excludes audio reading. GB = 10^9 bytes.

| Item | Default (1 GiB cache cap) | No cap (reference, see note) |
|---|---:|---:|
| RTFx (median per utterance) | 254.2 | 250.9 |
| RTFx (total audio / total processing time) | 256.1 | 255.9 |
| Peak RSS | 1.57 GB | 2.68 GB |
| mlx peak allocation (`mx.get_peak_memory()`) | 1.76 GB | 1.82 GB |
| macOS peak memory footprint | 3.19 GB | 14.40 GB |

- Only the uncapped footprint is from a same-condition rerun; other values are from the first run.
- Uncapped: median 0.050 seconds/utterance; loading 0.15 seconds with files in the OS cache.
- CLI, 33.8 minutes (2,026.5 seconds), 68 pieces: 6.97 seconds including startup/loading; maximum RSS 1.74 GB, peak footprint 3.48 GB.
- Published RTFx uses NVIDIA H200, batch 512; not comparable.
- Uncapped on the same 33.8-minute file: footprint 13.96 GB, active memory constant at 1.25 GB. Retained freed Metal buffers, not a leak; the default cap limits this growth.

## Conversion

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

- bf16 values/types unchanged. Only convolution axes move to channels-last: 2-D [O,I,H,W]→[O,H,W,I]; 1-D, including depthwise, [O,I,K]→[O,K,I].
- Repeated conversion gives bit-identical tensors. File SHA-256 may differ because safetensors metadata-key order is not fixed.

## Files

| File | Contents |
|---|---|
| `model.safetensors` | bf16; 1,231,827,296 bytes; SHA-256: `96600ed4ca8a4d9d45e6ed6bfd276ab9109da3f1c9e20e2bb0870e6f894fe5e4` |
| `config.json` | Model configuration |
| `tokenizer.model` | SentencePiece BPE, 3,072 tokens |
| `LICENSE` | License |
| `NOTICE` | Attribution |

## Limitations

- As with the original model, fillers ("えー", "あの") may be omitted; kanji/kana/numeral style cannot be selected.
- Silence/noise may produce short words. Four synthetic 10-second silence/weak-white-noise inputs produced none; real recordings untested.
- Splitting above 30 seconds raises CER. For 150 FLEURS utterances joined into 33.8 minutes, CER strict: 4.10% individually, 6.44% with ~30-second splits, 6.39% in PyTorch (float32). This reflects the model and splitting rule following official `transcribe_longform`, rather than MLX numerics.
- About 1 point comes from joining different speakers (5.08% even when grouping intact utterances into ≤30 seconds); this may be smaller with one continuous speaker. About 1.4 points come from mid-utterance cuts (50 of 67 boundaries).
- Batch 1 only. No streaming, speaker labels or official Haas-ASR custom vocabulary.
- Token timestamps in 80 ms steps, not word timestamps.

## License and attribution

- Weights/code: Apache-2.0. Source weights: [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0), Copyright Third Intelligence, Inc. Attribution: `NOTICE`. "Haas" identifies the source.
- References: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo), [third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR) (official inference package), [matsuurajun/haas-mlx](https://github.com/matsuurajun/haas-mlx), [Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx): Apache-2.0. [flash-linear-attention](https://github.com/fla-org/flash-linear-attention): MIT.
- Sanami's float32 ONNX is used only for numerical comparison; its weights are not redistributed. All distributed weights are unofficial conversions of the original `.nemo`.
- Evaluation data: FLEURS CC BY 4.0; JVNV CC BY-SA 4.0. Neither included nor redistributed.
