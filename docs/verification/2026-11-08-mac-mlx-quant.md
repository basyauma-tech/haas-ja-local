# MLX量子化版（8bit・4bit）の精度と速度（Mac、2026-11-08）

MLX版の線形層を8bitと4bitに量子化し、bf16版と同じ手順でasr-ja-bench v1.1のFLEURSとJVNVを測った。8bitはCER strictがFLEURS 3.68%、JVNV 4.49%で、公表値との差が0.01ポイントに収まり合格した。4bitはJVNVが5.00%で公表値を0.47ポイント上回り、不合格だった（合格基準は差0.3ポイント以内）。公開するのは8bit版だけとする。

量子化版はbf16版より小さく、メモリも少ないが、この機種では12〜14%遅かった（RTFx中央値で8bitが14.2%、4bitが12.3%低い）。

数値はすべて`var/bench/results/`以下のファイルから取った（`var/`はGit管理外）。環境はbf16版の記録（`2026-11-08-mac-mlx.md`）と同じ機種・OS・Python・mlx 0.32.4で、評価はコミット`138bab5`（作業ツリーに未コミットの変更なし）で行った。

## 量子化の方法

```sh
haas-ja-local convert-mlx --nemo var/upstream/Haas-ja-v1.0.nemo --out var/dist/mlx-8bit --quantize 8
haas-ja-local convert-mlx --nemo var/upstream/Haas-ja-v1.0.nemo --out var/dist/mlx-4bit --quantize 4
```

- `mlx.nn.quantize`のaffine量子化（group_size 64）を、すべての線形層に適用した。対象はエンコーダの前段の出力層・フィードフォワード・注意の射影（`linear_pos`を含む）、予測ネットワークの射影、結合ネットワークの3層。
- 畳み込み・正規化・埋め込み・`pos_bias_*`・`A_log`・`dt_bias`・BatchNormの統計・前処理の値は量子化せず、bf16版とビット単位で同じ値を保存した。
- 量子化した層は元のキー名の下に`weight`（uint32に詰めた値）・`scales`・`biases`（bf16）を持つ。`config.json`には`"quantization": {"bits": 8, "group_size": 64, "targets": "linear"}`を書く。
- 読み込み時は、エンコーダの`scales`・`biases`をbf16、予測ネットワークと結合ネットワークのものをfloat32にする。bf16版と同じ精度の方針である。
- 精度を合わせるための調整はしていない。1回目の変換と評価の数値をそのまま載せた。

| 版 | `model.safetensors`（バイト） | bf16版との比 |
|---|---:|---:|
| bf16 | 1,231,827,296 | 1.00 |
| 8bit | 727,881,324 | 0.59 |
| 4bit | 459,079,919 | 0.37 |

## 精度

```sh
uv run --extra eval --extra mlx python scripts/eval_bench.py --model var/dist/mlx-8bit --backend mlx \
  --bench var/bench/asr-ja-bench-v1.1 --subset fleurs jvnv --out var/bench/results/mlx-8bit
```

4bitは`mlx-4bit`に替えて同じコマンドで測った。bf16の値は`2026-11-08-mac-mlx.md`の測定である。

| 版 | サブセット | CER strict | CER flex | CER kana | 公表値strict | 差（ポイント） | 判定 |
|---|---|---:|---:|---:|---:|---:|---|
| bf16 | FLEURS | 3.65 | 2.98 | 2.33 | 3.65〜3.67 | 0.00 | 合格 |
| bf16 | JVNV | 4.51 | 3.06 | 2.25 | 4.50〜4.53 | 0.00 | 合格 |
| 8bit | FLEURS | 3.68 | 3.01 | 2.37 | 3.65〜3.67 | +0.01 | 合格 |
| 8bit | JVNV | 4.49 | 3.06 | 2.25 | 4.50〜4.53 | -0.01 | 合格 |
| 4bit | FLEURS | 3.70 | 3.01 | 2.40 | 3.65〜3.67 | +0.03 | 合格 |
| 4bit | JVNV | 5.00 | 3.58 | 2.62 | 4.50〜4.53 | +0.47 | 不合格 |

- 値は%。strictの小数第3位までは、8bitがFLEURS 3.683・JVNV 4.489、4bitがFLEURS 3.701・JVNV 5.001。
- 差は公表値の幅の近い端からの差で、幅の中なら0とした。
- 書き起こしがbf16版と文字列で一致した発話は、8bitがFLEURS 577/650（88.8%）・JVNV 1,168/1,375（84.9%）、4bitがFLEURS 368/650（56.6%）・JVNV 468/1,375（34.0%）。各版の`hyps.jsonl`を突き合わせた。
- 出典: `var/bench/results/mlx-8bit/`、`var/bench/results/mlx-4bit/`の`{fleurs,jvnv}/summary.json`と`hyps.jsonl`。

## 速度とメモリ

条件はbf16版の記録と同じ（バッチ1、1発話ずつ、先頭3発話をウォームアップとして除き、FLEURS 647発話で集計）。比較のため、bf16版も同じ時間帯に測り直した。メモリフットプリントを取るため、`uv run`を通さず`.venv/bin/python`を`/usr/bin/time -l`の下で直接実行した。

```sh
/usr/bin/time -l .venv/bin/python scripts/bench_speed.py --model var/dist/mlx-8bit --backend mlx \
  --bench var/bench/asr-ja-bench-v1.1 --subset fleurs --out var/bench/results/mlx-8bit-speed-t14.json
```

| 項目 | bf16 | 8bit | 4bit |
|---|---:|---:|---:|
| RTFx（1発話ごとの中央値） | 260.1 | 223.3 | 228.1 |
| RTFx（合計音声÷合計処理時間） | 263.4 | 226.6 | 232.8 |
| 1発話の処理時間（中央値） | 0.048秒 | 0.057秒 | 0.055秒 |
| ピークRSS（読み込み直後） | 1.57GB（1.30GB） | 1.09GB（0.80GB） | 0.82GB（0.53GB） |
| mlxのピーク割り当て（読み込み直後） | 1.79GB（1.26GB） | 1.23GB（0.74GB） | 0.97GB（0.47GB） |
| macOSのピークメモリフットプリント | 3.20GB | 2.63GB | 2.40GB |

- GBは10^9バイト。キャッシュ上限（1GiB）は既定のまま。
- 出典: `var/bench/results/mlx-{bf16,8bit,4bit}-speed-t14.json`、フットプリントは`var/bench/results/aux/mlx-{bf16,8bit,4bit}-speed-t14-time.txt`。
- 遅くなったのはエンコーダである。FLEURSの先頭103発話（ウォームアップ3発話を除く100発話、音声1,301秒）でencodeと復号を分けて測ると、encodeはbf16 1.85秒、8bit 2.41秒、4bit 2.37秒、復号は2.84秒、2.99秒、2.99秒だった（`aux/quant-split.json`、`aux/quant_split.py`）。量子化した行列積がbf16の行列積より遅い理由は調べていない。

## テスト（小さな乱数モデル）

`tests/test_mlx_quant.py`で、変換と読み込みの往復、キー集合と形状、量子化しない値のビット一致、CLIの引数を確かめた。線形層の入力次元を64の倍数にした小さな乱数モデルで、float32計算のbf16版と比べた相対誤差（最大の差÷基準の最大の絶対値、乱数mel 3本）は次のとおり。

| 版 | エンコーダ出力 | logits（3ステップ） |
|---|---:|---:|
| 8bit | 1.26e-2 | 2.18e-2 |
| 4bit | 1.90e-1 | 1.84e-1 |

- 逆量子化した重みをfloat32で持つモデルとは、どちらの版も相対誤差1e-5未満で一致した。差は量子化そのものによるもので、層の置き換え漏れや取り違えはない。
- 8bitはこの乱数モデルで3本の乱数melとも、greedy復号のトークン列がbf16版と一致した。

## 公開

- 8bit版: 公開用のファイルを`var/dist/publish/mlx-8bit/`にそろえた（`model.safetensors`・`config.json`・`tokenizer.model`・`LICENSE`・`NOTICE`）。モデルカードは後で作る。公開先は`basyauma-tech/Haas-ja-v1.0-mlx-8bit`。
- 4bit版: JVNVのCER strictが公表値を0.47ポイント上回り基準を外れたため、公開しない。

## 未測定

- asr-ja-benchの残りのサブセット（JSUT、TEDxJP-10K、SpeechBSD）。
- 量子化版での長時間音声（30秒を超える分割認識）と無音・雑音だけの入力。
- 量子化版とtorch（float32）のトークン列の一致。bf16版とは書き起こしの文字列だけを比べた。
- group_size 64以外（32・128）での精度。
- M5 Max以外の機種、メモリの少ない機種、バッチ2以上。
