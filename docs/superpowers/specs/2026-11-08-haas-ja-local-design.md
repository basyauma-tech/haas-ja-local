# Haas-ja-v1.0のMLX版・ONNX版を作成して公開する設計

日付: 2026-11-08。状態: 承認済み（2026-11-08）。実装計画は[plans/2026-11-08-haas-ja-local.md](../plans/2026-11-08-haas-ja-local.md)。利用者の決定: MLX版はMacで先行、ONNX版はWindows機で後続（同じセッションを引き継ぐ）。

## 1. 目的と背景

[third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)は日本語音声認識モデル（FastConformerエンコーダ＋TDTデコーダ、予測ネットワークはKimi Delta Attention）で、
配布物は`Haas-ja-v1.0.nemo`1本、公式の実行環境はLinux＋NVIDIA GPU＋NeMo 2.7に限られる。予測ネットワークがflash-linear-attentionのTritonカーネルに依存し、公式に「CPU and macOS are not supported」と明記されている。

このモデルをNeMo・CUDA・Tritonなしで動かせる形式に変換し、basyauma-technology名義でHugging Faceに公開する。

- MLX版: Apple SiliconのMetalで動く。
- ONNX版: ONNX RuntimeのCPUで動く（Apple Silicon・Linux・Windows）。

利用者は特定の社内用途ではなく、Hugging Faceで見つけた任意の人。モデルカードを読んだだけで動かせる状態を到達点とする。

## 2. 成果物

### 2.1 GitHubリポジトリ`basyauma-tech/haas-ja-local`（公開、Apache-2.0）

変換と推論のコードを1つのPythonパッケージ`haas_ja_local`にまとめる。重みは置かない。

| モジュール | 役割 |
|---|---|
| `haas_ja_local.audio` | 16kHz monoへの読み込み、NeMo互換のlog-mel前処理（numpy） |
| `haas_ja_local.torch_model` | 純PyTorchのリファレンス実装。`.nemo`から重みを読む。ONNX書き出しと数値検証の基準 |
| `haas_ja_local.mlx_model` | MLX実装。`model.safetensors`＋`config.json`を読む |
| `haas_ja_local.onnx_runtime` | ONNX Runtimeでエンコーダとデコーダを回す推論 |
| `haas_ja_local.decode` | TDT greedyデコード（両実行系で共通のロジック） |
| `haas_ja_local.longform` | 30秒超の音声を無音で分割して順に認識 |
| `haas_ja_local.convert` | `.nemo`→MLX重み、`.nemo`→ONNX（fp32）、ONNXのint8量子化 |
| `haas_ja_local.cli` | `haas-ja-local transcribe --backend mlx|onnx ...`、`haas-ja-local convert ...` |
| `scripts/` | 数値検証、ベンチマーク評価、速度計測、モデルカード用の数値出力 |

### 2.2 Hugging Face `basyauma-technology/Haas-ja-v1.0-mlx`

| ファイル | 内容 |
|---|---|
| `model.safetensors` | bfloat16。元チェックポイントの値を変えず、畳み込みの軸をMLXの配置に並べ替えたもの |
| `config.json` | 構成（語彙数、blank、duration候補、エンコーダ・予測ネットワークの寸法など） |
| `tokenizer.model` | 元の`.nemo`に含まれるSentencePieceモデルをそのまま |
| `README.md` | モデルカード（日本語・英語） |
| `LICENSE`、`NOTICE` | Apache-2.0と帰属表示 |

### 2.3 Hugging Face `basyauma-technology/Haas-ja-v1.0-onnx`

| ファイル | 内容 |
|---|---|
| `fp32/encoder.onnx`＋`fp32/encoder.onnx.data`、`fp32/decoder_joint.onnx` | 基準となるfloat32。エンコーダは約2.4GBでprotobufの上限（2GB）を超えるため、重みを外部データファイルに分けた2本構成 |
| `int8/encoder.onnx`、`int8/decoder_joint.onnx` | ONNX Runtimeの動的量子化（per-channel）。CERが合格ならCPU向け既定。細部は §12 |
| `config.json`、`tokenizer.model` | MLX版と同じ |
| `preprocessor.npz` | log-mel用の窓とメルフィルタバンク（チェックポイントのbf16をfloat32にしたもの）。ONNXグラフには前処理を含めないため別ファイルで配る |
| `README.md`、`LICENSE`、`NOTICE` | 同上 |

### 2.4ワークスペースの台帳

`~/workspace/basyauma/docs/repositories.md`に新リポジトリの行を追加する。

## 3. ゴール・非ゴール・失敗条件

ゴール

- `pip install git+https://github.com/basyauma-tech/haas-ja-local`と`hf download`の2手順で、Mac（MLX）と任意のCPU（ONNX）で日本語音声を認識できる。
- 精度が元モデルと同等であることを公開データで実測し、モデルカードに載せる。
- 変換を手元で再現できる（変換スクリプトと手順を公開する）。

非ゴール（今回は作らない）

- ストリーミング認識、ビームサーチ、カスタム語彙、話者分離。
- CoreML・iOS・Android向け、Linux CUDA向けの最適化。
- PyPIへの登録（gitインストールで足りる。後で追加できる）。
- basyauma-minutesへの組み込み（別タスク。同リポジトリの`docs/superpowers/specs/2026-11-08-haas-ja-eval-design.md`が第三者のONNX変換をMac経路の参考値に使う計画なので、本件のONNX版が完成したらそちらへ差し替える候補になる）。

失敗条件（動いても失敗とみなす）

- 公式・提供元と関係があるように読める表記。すべて「非公式の変換物」と明記する。
- 根拠のない精度・速度の数値。未測定は未測定と書く。
- 元モデルのApache-2.0の帰属表示（NOTICE）の欠落。
- 他者の変換物（重みファイル）の再配布。重みは必ず自分の手元で`.nemo`から変換する。

## 4. 制約

- 作業は2台に分ける。MLX版（torchリファレンス実装、MLX実装、変換、検証、計測、公開）はこのMac（Apple M5 Max、128GiB、macOS 27.2、Python 3.12）で行う。ONNX版（書き出し、int8量子化、x86での検証と計測、公開）は、利用者がこのセッションを引き継ぐWindows機で行う。
- そのため、パッケージはmacOSとWindowsの両方で動く構成にする。`mlx`はmacOS限定のoptional dependency、`onnxruntime`と`torch`は両OS共通のoptional dependencyにする。パスの扱いは`pathlib`で統一し、シェルスクリプトを置かない（PythonのCLIで済ませる）。
- 公式経路（Linux＋NVIDIA GPU）の出力を黄金データとして取る作業は、この設計に含めない。Windows機にNVIDIA GPUがある場合に試すかどうかは、そのときに利用者が決める。
- NeMoはインストールしない。`.nemo`はtarで、`model_weights.ckpt`（torchのstate dict）と`*_tokenizer.model`を含むので、torchだけで読める。
- 依存は最小にする。実行時: MLX版は`mlx`・`numpy`・`sentencepiece`・`soundfile`、ONNX版は`onnxruntime`・`numpy`・`sentencepiece`・`soundfile`。`torch`は変換と検証のときだけ（optional dependency）。
- 元モデルのApache-2.0に従い、NOTICEにThird Intelligenceの著作権表示を残す。参考にした公開実装（下記 §7）もNOTICEとREADMEに記載する。
- モデル名「Haas」は提供元の名称なので、リポジトリ名では元モデルの識別のために使うが、独自ブランドのように見せない。

## 5. アーキテクチャの要点（config.yamlより）

変換と実装はこの設定に従う。値は元リポジトリの`config.yaml`から転記した。

| 部位 | 設定 |
|---|---|
| 前処理 | 16kHz、n_fft 512、窓25ms（400点、hann）、ストライド10ms（160点）、128 mel、log、`per_feature`正規化。窓とフィルタバンクはチェックポイントにbf16で保存されている |
| エンコーダ | ConformerEncoder、24層、d_model 1024、8ヘッド、`rel_pos`、`dw_striding`サブサンプリング×8（256ch）、FF倍率4、conv kernel 9、batch_norm、`xscaling: true`、`untie_biases: true`、`use_bias: false`、pos_emb_max_len 5000 |
| 予測ネットワーク | `haas_asr.prednet.KDATransducerDecoder`、pred_hidden 640、2層、KDA 5ヘッド×64、expand_v 1.0、short conv 4、final normあり、`blank_as_pad: true` |
| 結合ネットワーク | RNNTJoint、joint_hidden 640、relu、出力3072＋blank(1)＋duration(5)＝3078 |
| デコード | TDT greedy、durations (0,1,2,3,4)、max_symbols 10。blankでduration 0のときは1進める |
| トークナイザ | SentencePiece BPE 3072（byte fallback）、blank id 3072 |

出力は80ms/フレーム（16kHz×160点×8）。トークンの時刻はフレーム番号×0.08秒で出す。

## 6. 検証計画（数値→一致→精度→速度の順）

1. 数値一致（torch基準の確立）: 自前のtorch実装の出力を、公開されているfp32 ONNX変換物（[Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx)の`fp32/`）と数発話で比較する。エンコーダ出力・結合ネットワークのlogitsの相対誤差1e-4未満を合格とする。これは精度の基準を手に入れるための比較で、相手の重みを再配布するものではない。
2. ONNX書き出し: torch実装からfp32を書き出し、ONNX Runtimeとtorchで同じ入力の出力を比較する。相対誤差1e-5未満。
3. MLX一致: MLX（bf16）とtorch（fp32）で、評価セットのトークン列の一致率を測る。bf16の丸めで一部は変わり得るので、一致率は参考値として記録し、合否は §6-4のCERで判断する。
4. 精度: [third-intelligence/asr-ja-bench](https://huggingface.co/datasets/third-intelligence/asr-ja-bench)v1.1のうち自動取得できるFLEURS（650発話）と、手動取得のJVNV（1,375発話）でCER strict / flex / kanaを[jacer](https://github.com/third-intelligence/jacer)で測る。公表値（FLEURS 3.65 / JVNV 4.50、strict）との差0.3ポイント以内を合格とする。JSUT・TEDxJP-10K・SpeechBSDは取得が重いため今回は対象外とし、未測定と明記する。測定対象はMLX bf16、ONNX fp32、ONNX int8の3つ。
   - 補足: 合格基準を外した系統が出たら、その系統の数値をそのまま載せて公開し、ONNX版の既定はfp32にする。int8については、デコーダ（KDAと結合ネットワーク）をfp32に残す選択的量子化を1回だけ試す。それでも外れるなら、その数値のまま「参考」として公開する。隠さない、捨てない、際限なく詰めない。
   - §6-1で第三者のfp32 ONNXと1e-4で一致しない場合、どちらが正しいかは手元では決められない。その場合は §6-1を参考値に格下げし、§6-4のCERが公表値と0.3ポイント以内なら実装を正とする。GPUで公式経路の出力（黄金データ）を取るかどうかは利用者の判断に委ねる（§12）。
5. 速度・メモリ: 同じ評価セットでRTFx（音声長÷処理時間）とピークメモリを測る。バッチ1、1発話ずつ。MLXはMetal、ONNXはCPU（スレッド数は4と全コアの2条件）。機種名と条件をカードに書く。公表のRTFx（H200、バッチ512）とは条件が違うことを明記する。
6. 長時間音声: 20分以上の録音1本で、分割認識が最後まで通ることとメモリが増え続けないことを確認する。
7. x86での検証: ONNX版の §6-2・§6-4・§6-5はWindows機（x86_64）で実機で行い、機種・OS・スレッド数をカードに書く。Mac上でのONNXの計測は行わない（行った場合は参考値と明記する）。
8. ユニットテスト: 乱数重みの小さな構成で、前処理・デコード・変換の往復をCI（GitHub Actions、macOSランナー）で回す。実モデルを使うテストは`MODEL_DIR`があるときだけ動く。

## 7. 参考にする公開実装

実装は自分で書くが、構造と落とし穴の確認に次を読む。flash-linear-attention以外はApache-2.0、flash-linear-attentionはMIT。NOTICEとREADMEに「参考にした」と記載する。コードを写す場合はそのファイルに出典を書く。

- NVIDIA NeMo: ConformerEncoder・RNNTJoint・TDT greedyの元定義。
- [third-intelligence/Haas-ASR](https://github.com/third-intelligence/Haas-ASR): KDA予測ネットワークの定義（`haas_asr.prednet`）と`transcribe_longform`の分割規則。
- flash-linear-attentionの`fla/ops/kda/naive.py`: KDAの再帰形の数式（ライセンスはMIT）。
- [matsuurajun/haas-mlx](https://github.com/matsuurajun/haas-mlx): MLXへの重みの並べ替え規則と、MLX実装での既知の注意点。
- [Sanami/haas-ja-onnx](https://huggingface.co/Sanami/haas-ja-onnx)の`scripts/`と`TECHNICAL.md`: ONNXの入出力仕様と純PyTorchのKDA。

## 8. ONNXの入出力仕様（決定）

NeMoのRNNT ONNX書き出しと同じ名前にし、既存のツール（sherpa-onnx系、既存のONNX推論スクリプト）から差し替えやすくする。

- エンコーダ: 入力`audio_signal` [B,128,T] f32、`length` [B] i64。出力`outputs` [B,1024,T']、`encoded_lengths` [B]。Tは16の倍数にパディングし、`length`には元の長さを渡す。
- デコーダ＋結合: 入力`encoder_outputs` [B,1024,1]、`targets` [B,1] i32、`target_length` [B] i32、`input_states_1`、`input_states_2`（各 [2,B,20480] f32）。出力`outputs` [B,1,1,3078]、`prednet_lengths`、`output_states_1`、`output_states_2`。
- `input_states_1`はKDAの再帰状態（5×64×64＝20480）、`input_states_2`はq/k/vのshort convバッファ（3×320×4＝3840を先頭に置き、残りは0）。層ごとに1行。状態の並び順はSanami版の公開スクリプト（`scripts/haas_onnx/decoder.py`）と同じにし、既存の推論スクリプトから差し替えられるようにする。
- opset 17。
- Bはバッチ次元として形は持たせるが、配布する推論コードはバッチ1だけを扱う。TDT greedyのバッチ対応（発話ごとに進む時刻が違う）は作らない。
- sherpa-onnxのNeMo transducer読み込みが参照するメタデータ（`vocab_size`、`pred_rnn_layers`、`pred_hidden`、`model_type`、`normalize_type`など）はONNXのmetadata_propsに入れておく。ただしsherpa-onnxで実際に動くことは要件にしない。試して動けばカードに書き、動かなければ「未確認」と書く。

## 9. モデルカードの構成（両リポジトリ共通、日本語→英語の順）

冒頭に「非公式の変換物で、Third Intelligenceとは無関係」と両言語で書く。YAML front matterは`license: apache-2.0`、`language: ja`、`base_model: third-intelligence/Haas-ja-v1.0`、`base_model_relation: quantized`（元モデルページの一覧に出るようにする）、`pipeline_tag: automatic-speech-recognition`、タグに`mlx`または`onnx`。

1. 概要（何を、どの形式に、どう変換したか。重みの値は変えていないこと）
2. 動作環境と必要なもの
3. 使い方（インストール2行、Python 5行、CLI 1行、長時間音声、タイムスタンプ）
4. 精度（§6-4の表。測定条件、未測定の明記、公表値の出典）
5. 速度とメモリ（§6-5の表。機種・OS・スレッド数）
6. 変換方法（コマンドと、変換元のリビジョン・SHA-256）
7. 制限事項（フィラーの脱落、無音で短い語が出ることがある、ストリーミング非対応、話者分離なし）
8. ライセンスと帰属（元モデル、参考実装、ベンチマークデータのライセンス）

## 10. 公開手順

1. GitHub: `gh repo create basyauma-tech/haas-ja-local --public`。mainへの直接pushは初回の土台だけとし、以降はPRで入れる。
2. Hugging Face: `hf repos create basyauma-technology/Haas-ja-v1.0-mlx --type model --public`（onnxも同様）、`hf upload`でファイルを上げる。アップロード前に手元で`hf download`→推論の往復を1度通す。
3. 公開後にモデルカードの表示（front matter、画像なし）と、元モデルページの一覧に出ることを確認する。
4. `docs/repositories.md`を更新する。

## 11. 作業の進め方

- 作業ディレクトリは`~/workspace/basyauma/haas-ja-local`。basyauma-minutesには触れない。
- 元の`.nemo`、評価データ、変換結果は`~/workspace/basyauma/haas-ja-local/var/`（gitignore）に置く。
- 第1段（Mac）: 土台（uv、ruff、pytest、CI）→ 前処理とトークナイザ → torch実装と数値一致 → MLX実装と変換 → 長時間分割とCLI → 評価と計測 → MLX版のモデルカード → GitHubとMLX版の公開 → `docs/repositories.md`更新。
- 第2段（Windows、同じセッションを引き継ぐ）: ONNX書き出し → int8量子化 → x86での数値一致・CER・速度の計測 → ONNX版のモデルカード → 公開。第1段でONNXの書き出しコードの骨組みと乱数重みのテストまではMacで作っておき、Windowsでは実モデルでの実行と検証から始められるようにする。
- 各段階で先に失敗するテストを書く。

## 12. 設計者の判断として決めたこと（違えば指摘を）

- 配布する推論コードはバッチ1のみ。RTFxもバッチ1で測る。
- ONNX版の入出力はNeMo互換の名前にし、状態の並びは既存の公開スクリプトに合わせる。sherpa-onnxで動くことは要件にしない。
- ONNX版のx86での検証と計測はWindows機で実機で行う。GitHub Actionsのubuntuランナーでの確認は、乱数重みの小さな構成のテストと、実モデルを使う数発話のトークン列一致（モデルの取得がCIの時間内に収まる場合）に限る。量子化の設定（activationの型、`reduce_range`）は実装者がx86の結果を見て決める。
- int8が精度基準を外れたら、数値を載せたまま公開し既定をfp32にする。選択的量子化は1回だけ試す。
- 公式経路（Linux GPU）で黄金データを取る作業は、この設計には含めない。必要になったら利用者に費用と時間を示して判断を仰ぐ。
- 公開の順序（利用者が決定）: MLX版とGitHubリポジトリを先に出し、ONNX版はWindows側の検証が済み次第出す。

- リポジトリ名: GitHub `basyauma-tech/haas-ja-local`、HF `Haas-ja-v1.0-mlx`・`Haas-ja-v1.0-onnx`。
- MLX版はbf16の1種類。ONNX版はfp32とint8（動的量子化）の2種類。fp16はCPUで遅いので出さない。
- 推論コードはHFリポジトリに同梱せず、GitHubからインストールする形にする。
- 精度測定はFLEURSとJVNVの2セット。
- モデルカードは1ファイルに日本語・英語を併記する。
- 他者の変換物は数値比較の基準としてだけ使い、再配布しない。

## 13. 実装者に任せること

- torch実装の内部構造（NeMoのモジュール分割に合わせるかどうか）。
- MLXでのbf16とfp32の使い分け（精度に影響しない範囲で）。
- 長時間分割の具体的な閾値（公式の`transcribe_longform`に近づける）。
- CLIの引数名と出力JSONの形。
- CIの構成と、実モデルを使うテストの切り替え方法。
