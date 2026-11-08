# haas-ja-local実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Haas-ja-v1.0（日本語ASR、FastConformer＋TDT＋KDA）をNeMo・CUDA・Tritonなしで動かすPythonパッケージ`haas_ja_local`を作り、MLX版（Mac）とONNX版（Windowsで後続）の重みをbasyauma-technology名義でHugging Faceに公開する。

**Architecture:** 純PyTorchのリファレンス実装が`.nemo`を読み、数値の基準とONNX書き出しの元になる。MLX実装とONNX Runtime実装は同じ`config.json`・同じ前処理（numpy）・同じTDT greedyデコード（純Python）を共有し、バックエンドは「エンコード」「予測ネットワーク1ステップ」「結合」の3操作だけを提供する。重みの名前は元チェックポイントのまま保ち、変換は軸の並べ替えだけにする。

**Tech Stack:** Python 3.12、uv、numpy、sentencepiece、soundfile、soxr（リサンプリング）。任意依存: mlx（macOSのみ）、onnxruntime、torch＋safetensors＋onnx（変換・検証）、jacer（評価）。ruff、pytest、GitHub Actions。

**Spec:** `docs/superpowers/specs/2026-11-08-haas-ja-local-design.md`

## Global Constraints

- Python `>=3.12,<3.13`。uvで管理し、`uv run pytest`・`uv run ruff check .`・`uv run ruff format --check .`を通す。
- 実行時の必須依存は`numpy`・`sentencepiece`・`soundfile`・`soxr`だけ。`mlx`は`sys_platform == 'darwin'`限定のextra、`onnxruntime`・`torch`は両OS共通のextra。
- macOSとWindowsの両方で動く。パスは`pathlib`、シェルスクリプトは置かない。
- 重みの値は変えない。MLX版はbf16のまま軸だけ並べ替える。ONNX版のfp32は`.nemo`のbf16をfloat32に上げたもの。
- 重みの名前は元チェックポイントのキー名を保つ（`encoder.layers.0.self_attn.linear_q.weight`など）。`num_batches_tracked`だけ捨てる。
- モデルセットのディレクトリは`config.json`・`tokenizer.model`と重みファイルで構成し、MLX版とONNX版で`config.json`の内容を同じにする。
- 日本語の文書・コメントでは数字・英字と日本語の間に半角スペースを入れない。
- 公開物には「非公式の変換物でThird Intelligenceとは無関係」と日英で明記し、Apache-2.0のNOTICEに元モデルの帰属を書く。他者の変換物（重み）は再配布しない。
- 元の`.nemo`: `var/upstream/Haas-ja-v1.0.nemo`、SHA-256 `c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa`（1,232,384,000バイト、2026-11-08取得）。`var/`はgit管理外。
- 元チェックポイントは755テンソル（bf16 731、int64 24）。bf16の要素数は615,871,520、`num_batches_tracked`（int64、各1要素）を含めると615,871,544。
- git push・GitHubリポジトリ作成・HFリポジトリ作成はTask 11の手順でのみ行う。

## Review Focus

仕様が明示しないが利用者が踏みやすい入力。各行のテストは所有するタスクに入れてある。

1. 48kHz・ステレオ・PCM16のWAVやFLACを渡す → 16kHz monoのfloat32に変換して認識できる（Task 3）。
2. 1フレーム未満（160サンプル未満）や完全無音の音声 → 例外ではなく空文字を返す（Task 6）。
3. mel長が16の倍数ちょうど、または16で割って余り1の音声 → 出力長の計算が一致し、末尾のトークンが落ちない（Task 3・Task 5）。
4. 30秒超の音声 → 無音で分割され、トークンの時刻が全体の先頭からの秒になる。メモリが長さの二乗で増えない（Task 6）。
5. macOS以外で`--backend mlx`、extra未インストールで`--backend onnx` → 何を入れればよいかを示すエラーで終了する（Task 6）。

---

## ファイル構成

```
haas-ja-local/
  pyproject.toml, README.md, LICENSE, NOTICE, .gitignore, .github/workflows/ci.yml
  src/haas_ja_local/
    __init__.py        公開API: load_model, transcribe, HaasConfig, Transcript, TimedToken
    config.py          HaasConfig（config.json / model_config.yaml の読み書き）
    nemo_checkpoint.py .nemo の展開（state dict、tokenizer、config、SHA-256）※torch必須
    audio.py           load_audio（soundfile＋soxr）
    features.py        log_mel（numpy、NeMo互換）
    tokenizer.py       Tokenizer（sentencepiece）
    decode.py          TDT greedy（純Python、Backendプロトコル）
    longform.py        無音分割と結合
    pipeline.py        load_model / transcribe（短い音声と長い音声の振り分け）。公開関数 transcribe と名前が衝突しないよう transcribe.py から改名
    torch_model.py     純PyTorch実装（Backend）
    mlx_model.py       MLX実装（Backend）
    onnx_model.py      ONNX Runtime実装（Backend）
    convert/__init__.py
    convert/mlx.py     .nemo → MLXディレクトリ
    convert/onnx.py    .nemo → ONNX fp32、int8量子化
    cli.py             haas-ja-local コマンド
  scripts/
    verify_numerics.py 第三者のfp32 ONNXとの数値比較（Task 7）
    rebuild_bench.py   asr-ja-benchの再構築呼び出し（Task 9）
    eval_bench.py      jacerでCER（Task 9）
    bench_speed.py     RTFxとピークメモリ（Task 9）
  tests/
    conftest.py        tiny_config、tiny_nemo（乱数重みの小さな.nemo）、real_model_dir（環境変数）
    test_config.py, test_nemo_checkpoint.py, test_audio.py, test_features.py,
    test_tokenizer.py, test_decode.py, test_torch_model.py, test_longform.py,
    test_transcribe.py, test_cli.py, test_convert_mlx.py, test_mlx_model.py,
    test_convert_onnx.py, test_onnx_model.py, test_real_model.py（環境変数ゲート）
  docs/
    superpowers/specs, superpowers/plans
    verification/2026-11-XX-mac-mlx.md（Task 7・9の記録）
    model-cards/mlx/README.md, model-cards/onnx/README.md（HFへ上げる原稿）
  var/（git管理外）upstream/, reference/, bench/, dist/
```

共通の型（`decode.py`・`transcribe.py`で定義し、全タスクで使う）:

```python
@dataclass(frozen=True)
class TimedToken:
    id: int
    frame: int       # 発話内のエンコーダフレーム番号
    duration: int    # TDTが予測したduration（0〜4）

class Backend(Protocol):
    config: HaasConfig
    window_np: np.ndarray   # チェックポイントの窓（400,）をfloat32にしたもの。log_melに渡す
    fb_np: np.ndarray       # メルフィルタバンク [128, 257] float32
    def encode(self, mel: np.ndarray, n_frames: int) -> Any: ...          # mel [128, T_pad] f32 → バックエンド固有のエンコーダ出力（有効長に切り詰め済み）
    def num_frames(self, encoded: Any) -> int: ...
    def init_state(self) -> Any: ...                                       # 予測ネットワークのゼロ状態
    def step(self, encoded: Any, t: int, token: int, state: Any) -> tuple[np.ndarray, Any]: ...
        # token を state に入れて予測ネットワークを1ステップ進め、フレーム t と結合した logits [3078] f32 と「token を入れた後の状態」を返す。
        # NeMo の decoder_joint ONNX と同じ1回呼び出しの形。blank のときは呼び出し側が新しい状態を捨てる。

@dataclass(frozen=True)
class Transcript:
    text: str
    tokens: tuple[TimedToken, ...]
    offsets: tuple[float, ...]   # tokensと同じ長さ。各トークンの開始秒（長時間分割のオフセット込み）
```

---

### Task 1: プロジェクトの土台

**Files:**
- Create: `pyproject.toml`, `src/haas_ja_local/__init__.py`, `tests/__init__.py`, `tests/test_package.py`, `.gitignore`, `LICENSE`, `NOTICE`, `README.md`, `.github/workflows/ci.yml`
- Create: `docs/`（既存の設計書と本計画はそのまま）

**Interfaces:**
- Produces: パッケージ名`haas-ja-local`、import名`haas_ja_local`、`__version__ = "0.1.0"`。extras: `mlx`、`onnx`、`convert`（torch, safetensors, onnx, onnxruntime）、`eval`（jacer git, huggingface_hub, pyarrow）、dev group（pytest, ruff）。スクリプト`haas-ja-local = "haas_ja_local.cli:main"`。

- [ ] **Step 1: 失敗するテストを書く** `tests/test_package.py`

```python
def test_version():
    import haas_ja_local
    assert haas_ja_local.__version__ == "0.1.0"
```

- [ ] **Step 2: `git init`し、`pyproject.toml`を書く**

`[project]` name `haas-ja-local`、requires-python `>=3.12,<3.13`、dependencies `numpy>=2.0`、`sentencepiece>=0.2`、`soundfile>=0.13`、`soxr>=0.5`。optional-dependencies: `mlx = ["mlx>=0.30; sys_platform == 'darwin'"]`、`onnx = ["onnxruntime>=1.20"]`、`convert = ["torch>=2.6", "safetensors>=0.5", "onnx>=1.17", "onnxruntime>=1.20"]`、`eval = ["jacer @ git+https://github.com/third-intelligence/jacer@v1.2.0", "huggingface_hub>=0.30", "pyarrow>=18"]`。dependency-groups dev: `pytest>=8`、`ruff>=0.8`。build-system `uv_build`。`[tool.ruff]` line-length 120、target py312、select `E,F,W,I,B,UP,SIM,RUF`、ignore `RUF001,RUF002,RUF003`。`[tool.pytest.ini_options]` testpaths `tests`、markers `real_model: needs HAAS_JA_MODEL_DIR / HAAS_JA_NEMO`。

- [ ] **Step 3: テストが失敗することを確認**

Run: `uv sync --extra mlx --extra onnx --extra convert --dev && uv run pytest tests/test_package.py -v`
Expected: FAIL（`__version__`がない、またはimportエラー）

- [ ] **Step 4: `src/haas_ja_local/__init__.py`に`__version__`を置き、`.gitignore`（`var/`、`.venv/`、`__pycache__/`、`dist/`、`*.onnx`、`*.safetensors`）、`LICENSE`（Apache-2.0全文）、`NOTICE`を書く**

NOTICEの内容: 本パッケージの著作権（basyauma-technology、2026）、元モデル`third-intelligence/Haas-ja-v1.0`（Third Intelligence, Inc.、Apache-2.0）、参考にした実装（NVIDIA NeMo、third-intelligence/Haas-ASR、matsuurajun/haas-mlx、Sanami/haas-ja-onnxはApache-2.0。fla-org/flash-linear-attentionはMIT）、「非公式・無関係」の一文（日英）。

- [ ] **Step 5: `README.md`の骨組み（日本語→英語。概要、非公式の明記、インストール、使い方は後のタスクで埋める見出しだけ）と`.github/workflows/ci.yml`を書く**

CI: matrix `macos-15`（arm64）・`ubuntu-latest`・`windows-latest`、`astral-sh/setup-uv`、`uv sync --extra mlx --extra onnx --extra convert --dev`（`eval` extraはjacerの辞書依存が重いのでCIでは入れない。Windows/ubuntuでは`mlx` extraは解決時に自動で外れる）、`uv run ruff check .`、`uv run ruff format --check .`、`uv run pytest -m "not real_model"`。

- [ ] **Step 6: テストとlintが通ることを確認**

Run: `uv run pytest -v && uv run ruff check . && uv run ruff format --check .`
Expected: 1 passed、All checks passed

- [ ] **Step 7: コミット**

```bash
git add -A && git commit -m "chore: scaffold haas-ja-local package"
```

---

### Task 2: 設定とチェックポイント読み込み、tinyモデルのfixture

**Files:**
- Create: `src/haas_ja_local/config.py`, `src/haas_ja_local/nemo_checkpoint.py`, `tests/conftest.py`, `tests/test_config.py`, `tests/test_nemo_checkpoint.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class HaasConfig:
    sample_rate: int = 16000; n_fft: int = 512; win_length: int = 400; hop_length: int = 160
    n_mels: int = 128; pad_to: int = 16
    d_model: int = 1024; n_layers: int = 24; n_heads: int = 8; ff_expansion: int = 4
    conv_kernel_size: int = 9; subsampling_factor: int = 8; subsampling_channels: int = 256
    pred_hidden: int = 640; pred_layers: int = 2
    kda_heads: int = 5; kda_head_dim: int = 64; kda_conv_size: int = 4
    joint_hidden: int = 640; vocab_size: int = 3072
    durations: tuple[int, ...] = (0, 1, 2, 3, 4); max_symbols: int = 10
    source_repo: str = "third-intelligence/Haas-ja-v1.0"; source_sha256: str = ""
    @property blank_id -> int (= vocab_size)
    @property frame_seconds -> float (= hop_length * subsampling_factor / sample_rate = 0.08)
    @property kda_state_width -> int (= max(kda_heads*kda_head_dim*kda_head_dim, 3*kda_heads*kda_head_dim*kda_conv_size) = 20480)
    def to_json(self, path: Path) -> None; @classmethod from_json(cls, path) -> HaasConfig
    @classmethod from_model_config_yaml(cls, text: str) -> HaasConfig   # .nemo内のmodel_config.yaml
```

```python
# nemo_checkpoint.py（torch必須。importは関数内）
@dataclass(frozen=True)
class NemoCheckpoint:
    config: HaasConfig; state_dict: dict[str, "torch.Tensor"]; tokenizer_model: bytes; sha256: str
def read_nemo(path: Path) -> NemoCheckpoint
def expected_keys(config: HaasConfig) -> frozenset[str]   # 元チェックポイントのキー名（num_batches_tracked含む）
```

- conftest fixtures: `tiny_config`（d_model 16, n_layers 2, n_heads 2, ff_expansion 2, conv_kernel_size 3, subsampling_channels 4, n_mels 8, pred_hidden 8, pred_layers 2, kda_heads 2, kda_head_dim 4, kda_conv_size 4, joint_hidden 8, vocab_size 16）、`tiny_tokenizer_model`（sentencepieceで`tests/data/tiny_corpus.txt`からvocab 16・byte_fallbackで学習したbytes、セッション内キャッシュ）、`tiny_state_dict(tiny_config, seed=0)`（`expected_keys`の全キーに対し形状通りの乱数bf16。BatchNormのrunning_varは正、`num_batches_tracked`はint64）、`tiny_nemo(tmp_path_factory)`（上記をtarにした`.nemo`のPath）、`real_nemo`／`real_model_dir`（環境変数`HAAS_JA_NEMO`／`HAAS_JA_MODEL_DIR`。無ければskip）。

- [ ] **Step 1: 失敗するテストを書く** `tests/test_config.py`

```python
def test_defaults_match_spec():
    c = HaasConfig()
    assert (c.blank_id, c.frame_seconds, c.kda_state_width) == (3072, 0.08, 20480)
    assert c.durations == (0, 1, 2, 3, 4) and c.max_symbols == 10

def test_json_roundtrip(tmp_path):
    c = HaasConfig(source_sha256="ab" * 32)
    c.to_json(tmp_path / "config.json")
    assert HaasConfig.from_json(tmp_path / "config.json") == c

def test_from_model_config_yaml_reads_upstream_values():
    text = (Path(__file__).parent / "data" / "model_config_excerpt.yaml").read_text()
    c = HaasConfig.from_model_config_yaml(text)
    assert (c.d_model, c.n_layers, c.kda_heads, c.kda_head_dim, c.kda_conv_size) == (1024, 24, 5, 64, 4)
```

`tests/data/model_config_excerpt.yaml`は`var/upstream/config.yaml`から`preprocessor`・`encoder`・`decoder`・`joint`・`decoding`の節だけを抜いたもの（vocabularyとlabelsのリストは省く）。

- [ ] **Step 2: 失敗を確認** Run: `uv run pytest tests/test_config.py -v` Expected: FAIL（import error）

- [ ] **Step 3: `config.py`を実装する**

YAMLは依存を増やさないため、`from_model_config_yaml`は必要なキーだけを行単位の正規表現で読む（`d_model:`、`n_layers:`、`n_heads:`、`conv_kernel_size:`、`subsampling_factor:`、`subsampling_conv_channels:`、`ff_expansion_factor:`、`pred_hidden:`、`pred_rnn_layers:`、`kda_num_heads:`、`kda_head_dim:`、`kda_conv_size:`、`joint_hidden:`、`vocab_size:`、`features:`、`n_fft:`、`max_symbols:`、`durations:`のリスト）。見つからないキーは例外にする。

- [ ] **Step 4: 通ることを確認** Run: `uv run pytest tests/test_config.py -v` Expected: PASS

- [ ] **Step 5: 失敗するテストを書く** `tests/test_nemo_checkpoint.py`

```python
def test_expected_keys_count_for_upstream_config():
    assert len(expected_keys(HaasConfig())) == 755

def test_read_tiny_nemo(tiny_nemo, tiny_config):
    ck = read_nemo(tiny_nemo)
    assert ck.config == tiny_config
    assert set(ck.state_dict) == expected_keys(tiny_config)
    assert ck.tokenizer_model[:2] != b"" and len(ck.sha256) == 64

@pytest.mark.real_model
def test_read_real_nemo(real_nemo):
    ck = read_nemo(real_nemo)
    assert ck.sha256 == "c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa"
    assert set(ck.state_dict) == expected_keys(ck.config)
    assert sum(v.numel() for v in ck.state_dict.values() if v.dtype != torch.int64) == 615871544
```

- [ ] **Step 6: 失敗を確認** Run: `uv run pytest tests/test_nemo_checkpoint.py -v` Expected: FAIL

- [ ] **Step 7: `conftest.py`のfixtureと`nemo_checkpoint.py`を実装する**

`expected_keys`は次の一覧を構成で展開する（形状はTask 5の表と同じ）: `preprocessor.featurizer.window`、`preprocessor.featurizer.fb`、`encoder.pre_encode.conv.{0,2,3,5,6}.{weight,bias}`、`encoder.pre_encode.out.{weight,bias}`、各層`encoder.layers.{i}.`配下の`norm_feed_forward1.{weight,bias}`、`feed_forward1.linear{1,2}.weight`、`norm_self_att.{weight,bias}`、`self_attn.{pos_bias_u,pos_bias_v}`、`self_attn.linear_{q,k,v,out,pos}.weight`、`norm_conv.{weight,bias}`、`conv.pointwise_conv1.weight`、`conv.depthwise_conv.weight`、`conv.batch_norm.{weight,bias,running_mean,running_var,num_batches_tracked}`、`conv.pointwise_conv2.weight`、`norm_feed_forward2.{weight,bias}`、`feed_forward2.linear{1,2}.weight`、`norm_out.{weight,bias}`；`decoder.prediction.embed.weight`、各層`decoder.prediction.dec_rnn.layers.{i}.`配下の`A_log`、`dt_bias`、`{q,k,v}_proj.weight`、`{q,k,v}_conv1d.weight`、`f_proj.{0,1}.weight`、`b_proj.weight`、`g_proj.{0,1}.weight`、`g_proj.1.bias`、`o_norm.weight`、`o_proj.weight`；`decoder.prediction.dec_rnn.norms.{i}.{weight,bias}`、`decoder.prediction.dec_rnn.final_norm.{weight,bias}`；`joint.pred.{weight,bias}`、`joint.enc.{weight,bias}`、`joint.joint_net.2.{weight,bias}`。

`read_nemo`は`tarfile`で`model_weights.ckpt`を`torch.load(map_location="cpu", weights_only=True)`、`*_tokenizer.model`をbytes、`model_config.yaml`を`HaasConfig.from_model_config_yaml`で読み、ファイル全体のSHA-256を`source_sha256`に入れる。

- [ ] **Step 8: 通ることを確認** Run: `uv run pytest tests/test_nemo_checkpoint.py -v`および`HAAS_JA_NEMO=var/upstream/Haas-ja-v1.0.nemo uv run pytest -m real_model tests/test_nemo_checkpoint.py -v` Expected: PASS

- [ ] **Step 9: コミット** `git commit -am "feat: config and .nemo checkpoint reader with tiny fixture"`

---

### Task 3: 音声読み込みとlog-mel前処理

**Files:**
- Create: `src/haas_ja_local/audio.py`, `src/haas_ja_local/features.py`, `tests/test_audio.py`, `tests/test_features.py`

**Interfaces:**
- Produces:

```python
def load_audio(path: Path, *, sample_rate: int = 16000) -> np.ndarray   # float32 1次元、[-1,1]、多chは平均、srが違えばsoxrで変換
def mel_frames(n_samples: int, config: HaasConfig) -> int                 # NeMo get_seq_len: n_samples // hop_length + 1
def log_mel(samples: np.ndarray, window: np.ndarray, fb: np.ndarray, config: HaasConfig) -> tuple[np.ndarray, int]
    # 戻り値: (mel [n_mels, T_pad] float32, n_frames)。T_padはpad_toの倍数（n_frames==0のときT_pad=0）
```

log_melの定義（NeMo `FilterbankFeatures`、推論時）: (1) pre-emphasis 0.97: `x[0]`はそのまま、`x[t] - 0.97*x[t-1]`。(2)両端に`n_fft//2`をreflectで詰める（`center=True`）。(3) hop 160でフレーム化、`window`（400点。checkpointのbf16をfloat32へ）を512点の中央に0埋めで配置して掛け、`rfft(512)`。(4)パワー（絶対値の2乗）に`fb`（[128,257]）を掛け、`log(x + 2**-24)`。(5)先頭`n_frames`フレームだけでmel次元ごとの平均と不偏標準偏差を取り、`(x - mean) / (std + 1e-5)`。`n_frames`より後ろは0。(6)時間軸を`pad_to`の倍数に0で埋める。

- [ ] **Step 1: 失敗するテストを書く** `tests/test_audio.py`

```python
def test_load_resamples_and_downmixes(tmp_path):
    sr = 48000; t = np.arange(sr) / sr
    stereo = np.stack([np.sin(2*np.pi*440*t), np.zeros_like(t)], axis=1).astype(np.float32)
    sf.write(tmp_path / "a.wav", stereo, sr, subtype="PCM_16")
    y = load_audio(tmp_path / "a.wav")
    assert y.dtype == np.float32 and y.ndim == 1 and abs(len(y) - 16000) <= 1
    assert 0.2 < np.abs(y).max() < 0.6   # 平均で振幅が半分になる
def test_load_flac_passthrough_16k(tmp_path): ...  # 16kHz monoのFLACがそのまま読める（長さ一致）
```

- [ ] **Step 2: 失敗を確認** Run: `uv run pytest tests/test_audio.py -v` Expected: FAIL

- [ ] **Step 3: `audio.py`を実装する**（`soundfile.read(dtype="float32", always_2d=True)` → 平均 → `soxr.resample(y, sr, 16000, quality="HQ")`）

- [ ] **Step 4: 通ることを確認** Run: `uv run pytest tests/test_audio.py -v` Expected: PASS

- [ ] **Step 5: 失敗するテストを書く** `tests/test_features.py`

```python
def _torch_reference(samples, window, fb, cfg):  # torch.stftでNeMoの式をそのまま書いた参照実装（テスト内に置く）
    ...
@pytest.mark.parametrize("n", [160*16, 160*16 + 1, 160*17 + 80, 16000*3 + 7])
def test_log_mel_matches_torch_reference(n, tiny_config):
    rng = np.random.default_rng(0); x = rng.standard_normal(n).astype(np.float32) * 0.1
    window = np.hanning(tiny_config.win_length + 1)[:-1].astype(np.float32)  # 値は何でもよい（同じ窓を両方に渡す）
    fb = rng.random((tiny_config.n_mels, tiny_config.n_fft // 2 + 1), dtype=np.float32)
    mel, n_frames = log_mel(x, window, fb, tiny_config)
    ref, ref_frames = _torch_reference(x, window, fb, tiny_config)
    assert n_frames == ref_frames == n // 160 + 1
    assert mel.shape[1] % 16 == 0 and mel.shape[1] >= n_frames
    np.testing.assert_allclose(mel[:, :n_frames], ref[:, :n_frames], rtol=1e-4, atol=1e-4)
    assert np.all(mel[:, n_frames:] == 0)

def test_log_mel_too_short_returns_empty(tiny_config):
    mel, n = log_mel(np.zeros(100, np.float32), ..., tiny_config)
    assert n == 1 or n == 0  # 100サンプル: 100//160+1 = 1フレーム。例外にならない
```

参照実装: `torch.stft(x, 512, 160, 400, window=torch.from_numpy(window), center=True, pad_mode="reflect", return_complex=True)`、`.abs()**2`、`fb @ power`、`log(+2**-24)`、`torch.std(unbiased=True)`で正規化、定数1e-5。これはNeMoの`FilterbankFeatures.forward`と同じ式。

- [ ] **Step 6: 失敗を確認** Run: `uv run pytest tests/test_features.py -v` Expected: FAIL

- [ ] **Step 7: `features.py`を実装する**（`np.lib.stride_tricks.sliding_window_view`でフレーム化、`np.fft.rfft`、float32）

- [ ] **Step 8: 通ることを確認** Run: `uv run pytest tests/test_features.py -v` Expected: PASS

- [ ] **Step 9: コミット** `git commit -am "feat: audio loading and NeMo-compatible log-mel features"`

---

### Task 4: トークナイザとTDT greedyデコード

**Files:**
- Create: `src/haas_ja_local/tokenizer.py`, `src/haas_ja_local/decode.py`, `tests/test_tokenizer.py`, `tests/test_decode.py`

**Interfaces:**
- Produces:

```python
class Tokenizer:
    def __init__(self, model: bytes | Path) -> None
    vocab_size: int
    def decode(self, ids: Sequence[int]) -> str
    def pieces(self, ids: Sequence[int]) -> list[str]
```

```python
# decode.py
TimedToken, Backend（冒頭の定義どおり）
def greedy_tdt(backend: Backend, encoded: Any) -> list[TimedToken]
```

greedy_tdtの規則（NeMoのTDT label-looping greedyと同じ）: `t = 0; state = init_state(); last = blank_id; symbols = 0`。`t < T`の間: `logits, new_state = step(encoded, t, last, state)`；`k = argmax(logits[:vocab_size+1])`、`d = durations[argmax(logits[vocab_size+1:])]`。`k == blank`なら`t += max(d, 1)`、`symbols = 0`、`last`と`state`は変えない（`new_state`は捨てる）。そうでなければ`TimedToken(k, t, d)`を追加し`last, state = k, new_state`、`t += d`、`d == 0`なら`symbols += 1`、`symbols >= max_symbols`なら`t += 1; symbols = 0`、`d > 0`なら`symbols = 0`。blankが続く間は同じ`(last, state)`で`step`を呼ぶので予測ネットワークの計算が重複するが、予測ネットワークは小さいので許容する。

- [ ] **Step 1: 失敗するテストを書く** `tests/test_tokenizer.py`

```python
def test_tokenizer_roundtrip(tiny_tokenizer_model):
    tok = Tokenizer(tiny_tokenizer_model)
    assert tok.vocab_size == 16
    ids = tok.encode("あい")  # テスト補助としてencodeも持つ（推論では使わない）
    assert tok.decode(ids) == "あい"
def test_decode_empty_is_empty_string(tiny_tokenizer_model):
    assert Tokenizer(tiny_tokenizer_model).decode([]) == ""
```

- [ ] **Step 2: 失敗を確認** → **Step 3: `tokenizer.py`を実装**（`spm.SentencePieceProcessor(model_proto=bytes)`）→ **Step 4: 通ることを確認**

- [ ] **Step 5: 失敗するテストを書く** `tests/test_decode.py`（台本どおりにlogitsを返す`ScriptedBackend`を定義。`step(encoded, t, token, state)`は`token`と`t`から辞書でlogitsを引き、状態は呼び出し回数を数えるタプルにする）

```python
def test_blank_with_duration_zero_advances_one_frame(): ...   # blank, d=0 → tが1進む、トークンなし
def test_token_with_duration_skips_frames(): ...               # token 5, d=3 → TimedToken(5, 0, 3)、次のtは3
def test_two_tokens_same_frame_then_blank(): ...               # d=0のトークン2つ → 同じframeで2つ出る
def test_max_symbols_forces_advance(tiny_config): ...          # 常にtoken, d=0を返す → 1フレームあたりmax_symbols個で止まりtが進む
def test_state_not_updated_on_blank(): ...                     # blankのとき次のstepに渡るtokenとstateが前と同じ（ScriptedBackendが記録する）
def test_empty_encoder_output_returns_no_tokens(): ...         # T=0 → []
```

- [ ] **Step 6: 失敗を確認** → **Step 7: `decode.py`を実装** → **Step 8: 通ることを確認** Run: `uv run pytest tests/test_decode.py tests/test_tokenizer.py -v` Expected: PASS

- [ ] **Step 9: コミット** `git commit -am "feat: tokenizer and TDT greedy decoding"`

---

### Task 5: 純PyTorchのリファレンス実装

**Files:**
- Create: `src/haas_ja_local/torch_model.py`, `tests/test_torch_model.py`

**Interfaces:**
- Consumes: `HaasConfig`、`read_nemo`、`Backend`
- Produces:

```python
class ConvSubsampling(nn.Module)        # キー: pre_encode.conv.{0,2,3,5,6}, pre_encode.out
class RelPositionalEncoding(nn.Module)  # パラメータなし。forward(x) -> (x * sqrt(d_model), pos_emb [1, 2T-1, d_model])
class RelPositionMultiHeadAttention(nn.Module)  # linear_q/k/v/out/pos (bias=False), pos_bias_u/v [H, d_k]
class ConformerFeedForward(nn.Module)   # linear1 → SiLU → linear2（bias=False）
class ConformerConvolution(nn.Module)   # pointwise_conv1(d→2d,k=1) → GLU(dim=1) → depthwise_conv(k=9,pad=4,groups=d) → batch_norm → SiLU → pointwise_conv2。全てbias=False
class ConformerLayer(nn.Module)         # 順序: norm_feed_forward1→ff1(×0.5)→norm_self_att→attn→norm_conv→conv→norm_feed_forward2→ff2(×0.5)→norm_out
class ConformerEncoder(nn.Module)       # forward(mel [B,128,T], length [B]) -> (enc [B,1024,T'], lengths [B])
class KimiDeltaAttention(nn.Module)     # 1ステップ再帰形。forward(x [B,1,H], rec [B,HV,V,K], conv (q,k,v各 [B,D,W])) -> (y [B,1,H], rec', conv')
class KDAPredictionNetwork(nn.Module)   # embed(3073,640)、layers、norms（pre-norm）、final_norm。forward(token [B], rec [L,B,20480], conv [L,B,20480]) -> (g [B,640], rec', conv')
class Joint(nn.Module)                  # enc(1024→640), pred(640→640), joint_net.2(640→3078)。forward(f [B,1024], g [B,640]) -> logits [B,3078] = joint_net.2(relu(enc(f) + pred(g)))
class HaasTorch(nn.Module, Backend)     # encoder, decoder(KDAPredictionNetwork), joint, buffers window/fb。step() は decoder → joint を1回で行う
def load_torch_model(nemo: Path, *, dtype: torch.dtype = torch.float32) -> tuple[HaasTorch, Tokenizer]
def encoded_length(n_frames: int, config: HaasConfig) -> int     # ceilをsubsamplingの段数（log2(8)=3）だけ繰り返す
```

- state dictのキーが元チェックポイントと完全一致するようモジュール名を付ける（`load_state_dict(strict=True)`）。`num_batches_tracked`は`BatchNorm1d`が持つので一致する。
- 相対位置: `pos = arange(T-1, -T, -1)`、偶数次元sin・奇数次元cos、`div_term = exp(arange(0,d,2) * -(ln(10000)/d))`。注意: `q_u = q + pos_bias_u`、`q_v = q + pos_bias_v`、`ac = q_u @ kᵀ`、`bd = rel_shift(q_v @ pᵀ)[..., :T]`、`scores = (ac + bd) / sqrt(d_k)`。rel_shiftは左に1列0を詰めて`view(B,H,2T,T)`し先頭行を落として`view(B,H,T,2T-1)`。
- 単一発話前提: サブサンプリング後に`encoded_length(n_frames)`で切り詰め、マスクは使わない。
- KDAの1ステップ（float32で計算）: `q,k,v = silu(short_conv(proj(x)))`（short_convは直近W=4入力のdepthwise因果畳み込み、bias無し、キャッシュ更新は`cat([cache[...,1:], x])`）；`q,k`をhead_dim 64ごとに`x / sqrt(sum(x²)+1e-6)`でL2正規化し、`q *= 64**-0.5`；`gk = -exp(A_log)[h] * softplus(f_proj.1(f_proj.0(x)) + dt_bias)`（[B,HV,K]）；`beta = sigmoid(b_proj(x))`（[B,HV]）；状態`S [B,HV,V,K]`に対し`S = S * exp(gk)[:, :, None, :]`、`v' = (v - (S * k[:, :, None, :]).sum(-1)) * beta[..., None]`、`S = S + v'[..., None] * k[:, :, None, :]`、`o = (S * q[:, :, None, :]).sum(-1)`；`gate = g_proj.1(g_proj.0(x))`を[B,HV,64]に；`y = rmsnorm(o; o_norm.weight, eps=1e-5) * sigmoid(gate)`；`o_proj(y.reshape(B,320))`。予測ネットワーク全体: `x = embed(token)`；各層`x = x + layer(norms[i](x))`；`final_norm`。
- 状態の詰め方（Backendの`init_state`は`(rec, conv)`各`[L,1,20480]`のゼロ）: rec行は`S.reshape(-1)`（HV,V,Kの順）、conv行は`q[D,W],k,v`の順にflattenして連結し残りを0。

- [ ] **Step 1: 失敗するテストを書く** `tests/test_torch_model.py`

```python
def test_state_dict_keys_match_checkpoint(tiny_config):
    model = HaasTorch(tiny_config)
    assert set(model.state_dict()) == expected_keys(tiny_config)

def test_loads_tiny_nemo_strict(tiny_nemo): load_torch_model(tiny_nemo)

@pytest.mark.parametrize("n_frames", [16, 17, 31, 32, 33, 160])
def test_encoder_output_length(tiny_nemo, n_frames, tiny_config):
    model, _ = load_torch_model(tiny_nemo)
    mel = torch.randn(1, tiny_config.n_mels, -(-n_frames // 16) * 16)
    enc, lengths = model.encoder(mel, torch.tensor([n_frames]))
    assert enc.shape == (1, tiny_config.d_model, encoded_length(n_frames, tiny_config)) and int(lengths[0]) == enc.shape[2]

def test_encoded_length_examples(): assert [encoded_length(n, HaasConfig()) for n in (1, 8, 9, 16, 17, 100)] == [1, 1, 2, 2, 3, 13]

def test_kda_step_sequence_matches_batched_naive(tiny_config):
    # 独立に書いた「全ステップをループで回す」素朴実装（テスト内）と、step を U 回呼んだ結果が一致する。
    # 乱数入力 U=5。rtol 1e-5。素朴実装は S を [HV,K,V] で持ち、fla の fused_recurrent_kda の式をそのまま書く。
    ...

def test_joint_formula(tiny_config): ...   # reluと線形の組み合わせを手計算（torch.nn.functionalで再計算）と比較

def test_backend_protocol_end_to_end(tiny_nemo):
    model, tok = load_torch_model(tiny_nemo)
    mel, n = log_mel(np.random.default_rng(0).standard_normal(16000).astype(np.float32), model.window_np, model.fb_np, model.config)
    tokens = greedy_tdt(model, model.encode(mel, n))
    assert all(0 <= t.id < model.config.vocab_size for t in tokens)
```

- [ ] **Step 2: 失敗を確認** Run: `uv run pytest tests/test_torch_model.py -v` Expected: FAIL

- [ ] **Step 3: `torch_model.py`を実装する**（モジュールごとに上の規約どおり。`HaasTorch.encode(mel, n_frames)`はnumpyを受けてテンソル`[T',1024]`を返し、`step`はnumpy `[3078]`と状態タプルを返す）

- [ ] **Step 4: 通ることを確認** Run: `uv run pytest tests/test_torch_model.py -v` Expected: PASS

- [ ] **Step 5: 実モデルで煙テスト（環境変数ゲート、`tests/test_real_model.py`に追加）**

```python
@pytest.mark.real_model
def test_real_torch_transcribes_fixture_clip(real_nemo):
    # tests/data/ja_short.flac: 自分で録音またはTTSで作った2〜5秒の日本語（著作権の問題がない音声）。期待文字列は tests/data/ja_short.txt
    model, tok = load_torch_model(real_nemo)
    text = tok.decode([t.id for t in greedy_tdt(model, model.encode(*log_mel(load_audio(CLIP), model.window_np, model.fb_np, model.config)))])
    assert text  # 非空。内容の一致は Task 7 の数値検証で見る
```

Run: `HAAS_JA_NEMO=var/upstream/Haas-ja-v1.0.nemo uv run pytest -m real_model -v` Expected: PASS（出力文字列をログに出す）

- [ ] **Step 6: コミット** `git commit -am "feat: pure PyTorch reference implementation of the Haas model"`

---

### Task 6: 長時間分割・transcribe API・CLI

**Files:**
- Create: `src/haas_ja_local/longform.py`, `src/haas_ja_local/transcribe.py`, `src/haas_ja_local/cli.py`, `tests/test_longform.py`, `tests/test_transcribe.py`, `tests/test_cli.py`
- Modify: `src/haas_ja_local/__init__.py`（公開API）

**Interfaces:**
- Produces:

```python
# longform.py（公式transcribe_longformと同じ規則）
def split_at_quiet_points(audio: np.ndarray, *, sample_rate=16000, chunk_seconds=30.0, search_seconds=2.0, window_seconds=0.1) -> list[tuple[int, int]]
# 規則: startからtarget=start+30s。残りが30s以下なら最後まで1断片。探索範囲low=max(start+15s, target-2s) .. high=min(len-window, target+2s)。
#       範囲内で0.1s窓のエネルギー（二乗和）最小の窓の中心をcutとする。範囲が成立しなければcut=target。

# transcribe.py
def load_model(model_dir_or_nemo: Path, *, backend: Literal["mlx", "onnx", "torch"] | None = None) -> tuple[Backend, Tokenizer]
#   backend None: ディレクトリにmodel.safetensorsがあればmlx、fp32/ またはint8/ があればonnx、.nemoならtorch。
#   必要なextraが無ければImportErrorを "pip install 'haas-ja-local[mlx]'" を含むメッセージで投げる。
def transcribe(backend: Backend, tokenizer: Tokenizer, audio: np.ndarray, *, longform_threshold_seconds: float = 30.0) -> Transcript
#   audioが閾値以下なら1回、超えればsplit_at_quiet_pointsで分割し、各断片のTimedTokenをoffsets=start/srを付けて連結。textはstripして "" で連結。
#   160サンプル未満またはn_frames==0ならTranscript("", (), ())。
```

CLI（`argparse`、`haas-ja-local`）:
- `transcribe MODEL AUDIO... [--backend mlx|onnx|torch] [--timestamps] [--json] [--threads N]`：既定は1行1ファイルでテキスト出力。`--json`で`{"file":..., "text":..., "tokens":[{"text","start","end"}]}`（`--timestamps`時のみtokens）。start=`offset + frame*0.08`、end=`start + max(duration,1)*0.08`。
- `convert-mlx --nemo PATH --out DIR`（Task 8）、`convert-onnx --nemo PATH --out DIR [--int8]`（Task 10）。

- [ ] **Step 1: 失敗するテストを書く** `tests/test_longform.py`

```python
def test_short_audio_is_single_chunk(): assert split_at_quiet_points(np.zeros(16000*20, np.float32)) == [(0, 16000*20)]
def test_cut_falls_in_quiet_gap():
    # 70秒: 29.0〜29.4秒だけ無音、他は乱数。最初の切れ目は 29.0〜29.4 秒の中に入る
    ...
def test_chunks_cover_audio_without_gaps_or_overlap(): ...  # 100秒乱数 → 断片が連続して全体を覆い、各断片は32秒以下、最後以外は15秒以上
```

`tests/test_transcribe.py`（tinyモデル、torchバックエンド）:

```python
def test_transcribe_too_short_returns_empty(tiny_nemo): ...     # 100サンプル → text == "" かつtokens == ()
def test_transcribe_silence_does_not_raise(tiny_nemo): ...      # 3秒ゼロ → Transcriptを返す
def test_longform_offsets_increase(tiny_nemo, monkeypatch):
    # longform_threshold_seconds=1.0 で 3.5 秒の乱数を渡す。断片ごとの offsets が単調非減少で、最後の断片の offset >= 1.0*? → 分割された断片数 >= 2 を transcribe の内部呼び出しを数えて確認
def test_load_model_missing_backend_message(tmp_path, monkeypatch):
    # mlx を import 不可にして model.safetensors を置いたディレクトリ → ImportError のメッセージに "haas-ja-local[mlx]" を含む
```

`tests/test_cli.py`: `main(["transcribe", str(tiny_nemo), str(wav), "--json", "--timestamps"])`が終了コード0でJSONを出し、`tokens`の`start <= end`であること。

- [ ] **Step 2: 失敗を確認** → **Step 3: 実装** → **Step 4: 通ることを確認** Run: `uv run pytest tests/test_longform.py tests/test_transcribe.py tests/test_cli.py -v` Expected: PASS

- [ ] **Step 5: `__init__.py`から`load_model`、`transcribe`、`load_audio`、`HaasConfig`、`Transcript`、`TimedToken`をexportし、READMEの「使い方」をPython5行とCLI1行で埋める**

- [ ] **Step 6: lintと全テスト** Run: `uv run ruff check . && uv run ruff format --check . && uv run pytest -v` Expected: PASS

- [ ] **Step 7: コミット** `git commit -am "feat: longform splitting, transcribe API and CLI"`

---

### Task 7: 実モデルの数値検証（Mac）

**Files:**
- Create: `scripts/verify_numerics.py`, `docs/verification/2026-11-XX-mac-numerics.md`
- Modify: `tests/test_real_model.py`

**Interfaces:**
- Consumes: `load_torch_model`、`log_mel`、`greedy_tdt`
- 基準: `var/reference/sanami-onnx/`に`hf download Sanami/haas-ja-onnx --include "fp32/*" "haas_ort.py" "tokenizer.model"`で取得した第三者のfp32 ONNX（再配布しない。`var/`はgit管理外）。
- 入力: `var/bench/clips/`に、FLEURS ja_jp testから5発話（Task 9の再構築前は`datasets`を使わず`hf download google/fleurs`で該当ファイルだけ取得してよい）と、自作の長め（20秒前後）の発話1本。

- [ ] **Step 1: `scripts/verify_numerics.py`を書く**

引数: `--nemo`、`--reference-dir`、`--clips DIR`、`--out REPORT.json`。処理: 各clipについて(a)自前の`log_mel`と参照の`haas_ort.logmel`の差（max abs）、(b)自前のtorchエンコーダ出力と参照`encoder.onnx`出力（同じ自前melを入力）の相対誤差`max|a-b| / max|b|`、(c)同じエンコーダ出力・同じトークン列（参照側のgreedyで得たもの）を与えたときの各ステップのjoint logitsの相対誤差、(d)自前greedyと参照greedyのトークン列一致、(e)テキスト。JSONとMarkdown表を出す。

- [ ] **Step 2: 実行する**

Run: `uv run --extra convert --extra onnx python scripts/verify_numerics.py --nemo var/upstream/Haas-ja-v1.0.nemo --reference-dir var/reference/sanami-onnx --clips var/bench/clips --out var/verify/numerics.json`
Expected: (a) < 1e-3、(b) < 1e-4、(c) < 1e-4、(d)全clipで一致。外れた場合は設計§6の補足どおり: まずmelの差を切り分け（窓・reflect・seq_len・正規化定数）、次にサブサンプリング（ReLUの位置、flattenの順）、相対位置（符号、rel_shift）、KDA（gkの符号、L2正規化のeps、状態の軸順）の順に二分探索する。1e-4に届かなくても(d)が一致しCERが合格なら「参考値」として記録して先へ進む。

- [ ] **Step 3: `tests/test_real_model.py`に固定クリップの期待トークン列を追加する**

Step 2で一致したclipのうち2本について、トークン列を`tests/data/expected_tokens.json`に保存し、`@pytest.mark.real_model`で自前torch実装の出力がそれと一致することを確認するテストを書く（以後の回帰検出用。音声ファイルはFLEURSのCC-BY-4.0に従い出典をdocsに書く。リポジトリに含めず`var/bench/clips`から読む）。

- [ ] **Step 4: 検証記録を書く** `docs/verification/2026-11-XX-mac-numerics.md`: 環境（機種・OS・Python・torch・onnxruntimeの版）、基準物のリビジョン、各clipの数値表、不一致があればその原因と対処。

- [ ] **Step 5: コミット** `git add scripts/verify_numerics.py docs/verification tests && git commit -m "test: verify torch reference against public fp32 ONNX export"`

---

### Task 8: MLX実装と変換

**Files:**
- Create: `src/haas_ja_local/mlx_model.py`, `src/haas_ja_local/convert/__init__.py`, `src/haas_ja_local/convert/mlx.py`, `tests/test_convert_mlx.py`, `tests/test_mlx_model.py`
- Modify: `src/haas_ja_local/cli.py`（`convert-mlx`）

**Interfaces:**
- Produces:

```python
# convert/mlx.py（torchとsafetensorsが必要。mlxは不要 → Windowsでも動く）
MLX_LAYOUT = {  # 元キーのパターン → 変換
  "conv2d": [O,I,H,W] → [O,H,W,I]（pre_encode.conv.0/2/3/5/6）,
  "conv1d": [O,I,K] → [O,K,I]（conv.pointwise_conv1/2、q/k/v_conv1d）,
  "depthwise_conv1d": [D,1,K] → [D,K,1]（conv.depthwise_conv）,
  "drop": num_batches_tracked,
  "direct": その他 }
def convert_nemo_to_mlx(nemo: Path, out_dir: Path) -> None
#   out_dir/model.safetensors（bf16、キー名は元のまま）、out_dir/config.json（source_sha256付き）、out_dir/tokenizer.model
#   fail-closed: 変換後のキー集合 == expected_keys - {num_batches_tracked} でなければ例外。
```

```python
# mlx_model.py（import mlxは関数内／モジュール先頭でtry）
class HaasMLX(Backend):
    @classmethod def load(cls, model_dir: Path) -> "HaasMLX"   # mx.load(model.safetensors) → tree_unflatten → nn.Module.load_weights(strict=True)
    # encode: mel → mx.array bf16 → エンコーダ（bf16、softmax は float32）→ [T',1024]（mx.eval 済み）
    # step: KDA 再帰は float32、joint は bf16 行列積 → logits を float32 の numpy で返す。1ステップを mx.compile した関数で回す
```

MLX側のモジュールも元キー名で構成する（`encoder.layers.0.self_attn.linear_q.weight`がそのまま`load_weights`で入る）。MLXの`nn.Conv2d`/`nn.Conv1d`は`groups`を使う（depthwise）。BatchNormは推論モード（running statsを使う）。

- [ ] **Step 1: 失敗するテストを書く** `tests/test_convert_mlx.py`

```python
def test_convert_tiny_nemo_writes_expected_files(tiny_nemo, tmp_path):
    convert_nemo_to_mlx(tiny_nemo, tmp_path / "mlx")
    assert {p.name for p in (tmp_path / "mlx").iterdir()} == {"model.safetensors", "config.json", "tokenizer.model"}
    cfg = HaasConfig.from_json(tmp_path / "mlx" / "config.json"); assert len(cfg.source_sha256) == 64
def test_converted_tensors_have_mlx_layout_and_same_values(tiny_nemo, tmp_path):
    # safetensors.torch.load_file で読み、conv2d のキーが [O,H,W,I]、conv1d が [O,K,I]、depthwise が [D,K,1] で、permute を戻すと元と bit 一致（bf16 のまま）
def test_convert_fails_closed_on_unexpected_key(tiny_nemo, tmp_path, monkeypatch): ...  # state dictに余計なキーを足す → 例外
```

- [ ] **Step 2: 失敗を確認** → **Step 3: `convert/mlx.py`と`cli.py`の`convert-mlx`を実装** → **Step 4: 通ることを確認**

- [ ] **Step 5: 失敗するテストを書く** `tests/test_mlx_model.py`（`pytest.importorskip("mlx")`）

```python
def test_mlx_matches_torch_on_tiny_model(tiny_nemo, tmp_path):
    # tiny を MLX に変換し、同じ乱数 mel（1秒分）で encode → step(t=0, token=blank) の logits を torch(float32) と比較。
    # 比較のため MLX 側は float32 で計算する load オプション（compute_dtype=mx.float32）を持たせる。rtol 1e-3, atol 1e-3。
def test_mlx_step_state_roundtrip(tiny_nemo, tmp_path): ...  # 2ステップ進めた状態がtorchの状態（同じ詰め方）と一致（float32、rtol 1e-4）
def test_mlx_greedy_tokens_equal_torch_on_tiny(tiny_nemo, tmp_path): ...  # float32同士なら同じトークン列
```

- [ ] **Step 6: 失敗を確認** → **Step 7: `mlx_model.py`を実装** → **Step 8: 通ることを確認** Run: `uv run pytest tests/test_mlx_model.py tests/test_convert_mlx.py -v`

- [ ] **Step 9: 実モデルで変換と一致率（`tests/test_real_model.py`に追加、`real_model`マーク）**

Run: `uv run haas-ja-local convert-mlx --nemo var/upstream/Haas-ja-v1.0.nemo --out var/dist/mlx` → `ls -la var/dist/mlx`（model.safetensorsが約1.23GB）→ `HAAS_JA_MODEL_DIR=var/dist/mlx uv run pytest -m real_model -v`
テスト: Task 7の2本のclipで、MLX bf16のトークン列が期待トークン列と一致する（bf16で変わる場合は、テキストの文字一致率が0.98以上で合格とし、その旨をテスト名とコメントに書く）。

- [ ] **Step 10: コミット** `git commit -am "feat: MLX backend and .nemo to MLX converter"`

---

### Task 9: 評価と計測（Mac、MLX版）

**Files:**
- Create: `scripts/rebuild_bench.py`, `scripts/eval_bench.py`, `scripts/bench_speed.py`, `docs/verification/2026-11-XX-mac-mlx.md`

**Interfaces:**
- `scripts/rebuild_bench.py --subset fleurs jvnv --jvnv-zip PATH --out var/bench/asr-ja-bench-v1.1`: `hf download third-intelligence/asr-ja-bench`で`rebuild.py`とメタデータを取り、`rebuild.py`を`subprocess`で呼ぶ（`--subset fleurs`は自動取得、`--jvnv`はzip）。JVNVのzipは利用者が公式サイトから`var/bench/jvnv_ver1.zip`に置く。
- `scripts/eval_bench.py --model DIR --backend mlx --bench DIR --subset fleurs jvnv --out var/bench/results/<name>/`: `<subset>.jsonl`（`audio_filepath`または`audio`、`text`）を読み、1発話ずつ`transcribe`し、`jacer.cer(refs, hyps, jacer.METRICS)`でstrict/flex/kana（%）を出す。`hyps.jsonl`（id, reference, hypothesis）と`summary.json`（CER3種、発話数、音声合計秒、処理合計秒、RTFx、`jacer.METRICS_VERSION`、環境）を書く。
- `scripts/bench_speed.py --model DIR --backend mlx --bench DIR --subset fleurs`: 1発話ずつの処理時間（モデル読み込み後、ウォームアップ3発話を除く）からRTFx中央値・合計RTFx、`resource.getrusage`のmaxrss（macOSはbytes）、`mx.get_peak_memory()`を出す。

- [ ] **Step 1: 3本のスクリプトを書く**（評価と速度はバックエンド非依存。`--backend torch`でも動く）

- [ ] **Step 2: ベンチマークを再構築する**

Run: `uv run --extra eval python scripts/rebuild_bench.py --subset fleurs jvnv --jvnv-zip var/bench/jvnv_ver1.zip --out var/bench/asr-ja-bench-v1.1`
Expected: `fleurs.jsonl`650行、`jvnv.jsonl`1,375行。ハッシュ不一致で止まった場合は原因（zipの版）を記録して対処。

- [ ] **Step 3: 精度を測る**

Run: `uv run --extra eval --extra mlx python scripts/eval_bench.py --model var/dist/mlx --backend mlx --bench var/bench/asr-ja-bench-v1.1 --subset fleurs jvnv --out var/bench/results/mlx-bf16`
Expected: CER strictがFLEURS 3.65±0.3、JVNV 4.50±0.3。外れたら設計§6-4の補足に従う（数値はそのまま載せる）。

- [ ] **Step 4: 速度を測る**

Run: `uv run --extra mlx python scripts/bench_speed.py --model var/dist/mlx --backend mlx --bench var/bench/asr-ja-bench-v1.1 --subset fleurs --out var/bench/results/mlx-bf16-speed.json`
さらに20分以上の音声1本（`var/bench/long/`。自作または許諾のある音源）で`haas-ja-local transcribe --timestamps --json`を`/usr/bin/time -l`で実行し、所要秒とピークRSSを記録する。

- [ ] **Step 5: 検証記録を書く** `docs/verification/2026-11-XX-mac-mlx.md`: 環境表、コマンド、CER表（公表値との差）、速度表（バッチ1、Metal、bf16、ウォームアップ除外の条件）、長時間音声の結果、未測定（JSUT・TEDxJP-10K・SpeechBSD、x86、バッチ>1）。

- [ ] **Step 6: コミット** `git add scripts docs/verification && git commit -m "eval: asr-ja-bench scoring and speed benchmark for the MLX backend"`

---

### Task 10: ONNX書き出しと量子化のコード（Mac。tinyモデルでテストまで）

**Files:**
- Create: `src/haas_ja_local/convert/onnx.py`, `src/haas_ja_local/onnx_model.py`, `tests/test_convert_onnx.py`, `tests/test_onnx_model.py`
- Modify: `src/haas_ja_local/cli.py`（`convert-onnx`）

**Interfaces:**
- Produces:

```python
# convert/onnx.py（torch, onnx, onnxruntime）
class EncoderWrapper(nn.Module)       # forward(audio_signal [B,128,T], length [B] i64) -> (outputs [B,1024,T'], encoded_lengths [B] i64)
class DecoderJointWrapper(nn.Module)  # forward(encoder_outputs [B,1024,1], targets [B,1] i32, target_length [B] i32, input_states_1 [L,B,20480], input_states_2 [L,B,20480])
                                      #   -> (outputs [B,1,1,3078], prednet_lengths [B] i32, output_states_1, output_states_2)
def export_onnx(nemo: Path, out_dir: Path) -> None
#   out_dir/fp32/encoder.onnx (+ encoder.onnx.data、全テンソル1ファイル、size_threshold=1024)、out_dir/fp32/decoder_joint.onnx、config.json、tokenizer.model、preprocessor.npz（window と fb の float32。ONNX には前処理が含まれないため）
#   opset 17、dynamic_axes: audio_signal {0:"B",2:"T"}、length {0:"B"}、outputs {0:"B",2:"T_enc"}、encoder_outputs {0:"B"}、targets/target_length/states {1 or 0:"B"}
#   metadata_props: vocab_size=3072, blank_id=3072, model_type=tdt, durations="0,1,2,3,4", pred_rnn_layers=2, pred_hidden=640, normalize_type=per_feature,
#                   subsampling_factor=8, feat_dim=128, frame_seconds=0.08, source_repo, source_sha256, converter="haas-ja-local <version>", note="unofficial conversion"
def quantize_int8(fp32_dir: Path, out_dir: Path, *, exclude_decoder: bool = False) -> None
#   onnxruntime.quantization.quantize_dynamic(weight_type=QInt8, per_channel=True, reduce_range=<Task 12で決定>)をエンコーダとデコーダに。exclude_decoder=Trueならデコーダはfp32をコピー。
```

```python
# onnx_model.py
class HaasONNX(Backend):
    @classmethod def load(cls, model_dir: Path, *, variant: str | None = None, threads: int | None = None) -> "HaasONNX"
    # variant None: int8/ があれば int8、無ければ fp32。CPUExecutionProvider、ORT_ENABLE_ALL。
    # encode: audio_signal=mel[None], length=[n_frames] → outputs[0, :, :encoded_lengths[0]].T → [T',1024] f32
    # step: decoder_joint セッションを1回実行（encoder_outputs=encoded[t][None,:,None], targets=[[token]], target_length=[1], input_states_1/2=state）
    #       → (outputs[0,0,0], (output_states_1, output_states_2))
```

- [ ] **Step 1: 失敗するテストを書く** `tests/test_convert_onnx.py`（`pytest.importorskip("onnxruntime")`）

```python
def test_export_tiny_writes_files_and_metadata(tiny_nemo, tmp_path):
    export_onnx(tiny_nemo, tmp_path)
    m = onnx.load(tmp_path / "fp32" / "encoder.onnx", load_external_data=False)
    assert [i.name for i in m.graph.input] == ["audio_signal", "length"]
    d = onnx.load(tmp_path / "fp32" / "decoder_joint.onnx")
    assert [i.name for i in d.graph.input] == ["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"]
    assert {p.key: p.value for p in d.metadata_props}["model_type"] == "tdt"
def test_onnx_encoder_matches_torch_on_tiny(tiny_nemo, tmp_path): ...   # 同じmel → rtol 1e-4
def test_onnx_decoder_step_matches_torch_on_tiny(tiny_nemo, tmp_path): ...  # 3ステップ分のlogitsと状態が一致rtol 1e-4
def test_quantize_int8_runs_on_tiny(tiny_nemo, tmp_path): ...  # int8/ が出来てHaasONNX.load(variant="int8")でencodeできる
```

`tests/test_onnx_model.py`: `HaasONNX`を`Backend`として`greedy_tdt`に渡し、tinyモデルでtorchと同じトークン列になること。

- [ ] **Step 2: 失敗を確認** → **Step 3: `convert/onnx.py`、`onnx_model.py`、`convert-onnx`を実装** → **Step 4: 通ることを確認** Run: `uv run pytest tests/test_convert_onnx.py tests/test_onnx_model.py -v`

- [ ] **Step 5: 全テスト・lint・CI（3 OS）が緑であることを確認し、コミット** `git commit -am "feat: ONNX export, int8 quantization and ONNX Runtime backend"`

---

### Task 11: MLX版の公開（モデルカード、GitHub、Hugging Face、台帳）

**Files:**
- Create: `docs/model-cards/mlx/README.md`, `docs/model-cards/onnx/README.md`（骨組み。数値はTask 13）
- Modify: `README.md`, ワークスペース直下の`docs/repositories.md`（このリポジトリの外）

- [ ] **Step 1: MLX版のモデルカードを書く** `docs/model-cards/mlx/README.md`

front matter: `license: apache-2.0`、`language: [ja]`、`library_name: mlx`、`pipeline_tag: automatic-speech-recognition`、`base_model: third-intelligence/Haas-ja-v1.0`、`base_model_relation: quantized`、`tags: [mlx, apple-silicon, japanese, speech-recognition, parakeet, tdt, fastconformer]`。本文は設計§9の8節を日本語→英語の順で。精度・速度はTask 9の記録から転記し、条件（機種、Metal、bf16、バッチ1、asr-ja-bench v1.1、jacer v1.2）と未測定を明記。「非公式でThird Intelligenceとは無関係」を冒頭に日英で。変換元のリビジョン・SHA-256・変換コマンドを書く。

- [ ] **Step 2: 公開前の往復確認**

`var/dist/mlx`に`README.md`（上記）、`LICENSE`、`NOTICE`をコピーし、別の空ディレクトリに`cp -r`してから`uv run haas-ja-local transcribe <そのディレクトリ> var/bench/clips/<1本>`で認識できることを確認する。

- [ ] **Step 3: GitHubへ公開**

```bash
gh repo create basyauma-tech/haas-ja-local --public --description "Unofficial MLX / ONNX conversions and NeMo-free inference for Haas-ja-v1.0 (Japanese ASR)" --source . --remote origin --push
```
push後にCIが3 OSで緑になることを確認する。以後の変更はブランチ＋PRで入れる。

- [ ] **Step 4: Hugging Faceへ公開**

```bash
hf repos create basyauma-technology/Haas-ja-v1.0-mlx --type model --public
hf upload basyauma-technology/Haas-ja-v1.0-mlx var/dist/mlx . --commit-message "Add unofficial MLX conversion of third-intelligence/Haas-ja-v1.0"
```
公開ページでfront matterの表示、`base_model`のリンク、元モデルページの一覧への掲載を確認する。`hf download basyauma-technology/Haas-ja-v1.0-mlx --local-dir /tmp/x`→`transcribe`の往復を1回行う。

- [ ] **Step 5: READMEに公開先リンクを入れ、`docs/repositories.md`に行を追加し、コミット**

`| haas-ja-local/ | リポジトリ | https://github.com/basyauma-tech/haas-ja-local.git | main（2026-11-XX作成。Haas-ja-v1.0のMLX/ONNX変換と推論。HF: basyauma-technology/Haas-ja-v1.0-mlx） |`

```bash
git commit -am "docs: MLX model card and publication links"
```

---

### Task 12: ONNX版の実モデル検証と計測（Windows、x86_64）

前提: Windows機にPython 3.12とuv、`git clone https://github.com/basyauma-tech/haas-ja-local`、`var/upstream/Haas-ja-v1.0.nemo`（SHA-256を照合）、Task 9で再構築した`var/bench/asr-ja-bench-v1.1`（Macからコピーするか再構築）。

- [ ] **Step 1: 書き出す** Run: `uv run --extra convert haas-ja-local convert-onnx --nemo var/upstream/Haas-ja-v1.0.nemo --out var/dist/onnx`
Expected: `fp32/encoder.onnx`＋`encoder.onnx.data`（合計約2.4GB）、`fp32/decoder_joint.onnx`、`config.json`、`tokenizer.model`。

- [ ] **Step 2: 数値一致** Run: `uv run --extra convert --extra onnx python scripts/verify_numerics.py --nemo ... --onnx-dir var/dist/onnx/fp32 --clips var/bench/clips --out var/verify/onnx-fp32.json`（`verify_numerics.py`に`--onnx-dir`で自前ONNXを基準に比較するモードを足す。torch fp32との相対誤差1e-5未満、トークン列一致）。

- [ ] **Step 3: int8を作って精度を測る**

Run: `uv run --extra convert haas-ja-local convert-onnx --nemo ... --out var/dist/onnx --int8` → `eval_bench.py --backend onnx --variant fp32`と`--variant int8`をFLEURSとJVNVで。
判定: int8のCER strictが公表値±0.3以内なら既定をint8にする。外れたら`--int8 --exclude-decoder`で1回だけ再試行し、それでも外れたら既定をfp32にして数値はそのまま載せる。`reduce_range`は、x86でint8の結果がfp32から大きく外れる（CER差0.3超）場合にTrueを試す。決めた設定を`convert/onnx.py`の既定値にしてコミットする。

- [ ] **Step 4: 速度** Run: `bench_speed.py --backend onnx --variant int8 --threads 4`と`--threads <物理コア数>`、fp32も同様。機種・CPU名・OS・onnxruntimeの版を記録。

- [ ] **Step 5: 検証記録** `docs/verification/2026-11-XX-windows-onnx.md`（環境、コマンド、数値一致、CER表、速度表、量子化設定の決定理由、未測定）。

- [ ] **Step 6: ブランチでコミットし、PRを作る** `git checkout -b onnx-verification && git commit -am "eval: ONNX fp32/int8 verification on x86_64" && gh pr create --fill`

---

### Task 13: ONNX版の公開

- [ ] **Step 1: `docs/model-cards/onnx/README.md`を書く**（Task 11と同じ構成。`library_name: onnx`、`tags: [onnx, onnxruntime, cpu, japanese, speech-recognition, parakeet, tdt, fastconformer]`。入出力仕様（設計§8）と、fp32・int8の使い分け、x86とApple Siliconのどちらで測ったかを明記）

- [ ] **Step 2: `var/dist/onnx`に`README.md`・`LICENSE`・`NOTICE`をコピーし、別ディレクトリへのコピーで往復確認（`--backend onnx`、fp32とint8）**

- [ ] **Step 3: 公開**

```bash
hf repos create basyauma-technology/Haas-ja-v1.0-onnx --type model --public
hf upload basyauma-technology/Haas-ja-v1.0-onnx var/dist/onnx . --commit-message "Add unofficial ONNX (fp32/int8) conversion of third-intelligence/Haas-ja-v1.0"
```
`hf download`→`transcribe`の往復、ページ表示、元モデルページの一覧掲載を確認する。

- [ ] **Step 4: README・`docs/repositories.md`にONNX版のリンクを追記し、PRをマージして記録を残す**

---

## 自己レビュー（計画作成時）

- 仕様の§2（成果物）: Task 1・8・10・11・13。§5（構成）: Task 2・5。§6-1〜8: Task 7・10・8・9・9・6・12・1。§8: Task 10。§9: Task 11・13。§10: Task 11・13。§2.4: Task 11。
- 型の一貫性: `Backend`の4メソッド名（encode, num_frames, init_state, step）、`TimedToken(id, frame, duration)`、`Transcript(text, tokens, offsets)`、`HaasConfig`のフィールド名、`expected_keys`、`load_torch_model`／`HaasMLX.load`／`HaasONNX.load`を全タスクで統一した。
- Review Focusの5項目はTask 3・5・6のテストに入れた。

---

## 改訂（2026-11-08、ユーザー決定）: ONNX版と MLX 量子化版も Mac で開発する

利用者の指示により、x86 での確認は後回しにし、ONNX 版（fp32 / int8）と MLX 量子化版（8bit / 4bit）を含む全開発をこの Mac で行う。公開文書は本文で組織名に言及せず、要点だけを短く書き、文章は Codex CLI に書かせる。

### Task 12'（Mac）: ONNX 版の実モデル検証と計測

- 書き出し: `uv run --extra convert haas-ja-local convert-onnx --nemo var/upstream/Haas-ja-v1.0.nemo --out var/dist/onnx --int8`（fp32 と int8。`var/dist/onnx-mac-check` は参考確認用なので使わず、作り直す）。
- 数値一致: `scripts/verify_numerics.py --onnx-dir var/dist/onnx/fp32 ...`（torch と 1e-5、6 クリップのトークン一致、フレーム数一致のゲート込み）。
- 精度: `scripts/eval_bench.py --backend onnx --variant fp32|int8` を FLEURS と JVNV で。int8 が公表値±0.3 を外れたら `--int8 --exclude-decoder` を 1 回試し、それでも外れたら数値を載せて既定 fp32 のまま（既定は現状 fp32）。int8 が合格したら `HaasONNX.load` の既定を int8 に変えるかはカードに「既定は fp32、int8 は任意」と書く形で据え置く（x86 未確認のため）。
- 速度: `scripts/bench_speed.py --backend onnx --variant fp32|int8 --threads 4|<物理コア数>`（Apple Silicon CPU。x86 は未計測と明記）。
- 検証記録 `docs/verification/2026-11-08-mac-onnx.md`。

### Task 13'（Mac）: ONNX 版の公開

- カード `docs/model-cards/onnx/README.md` の TODO を実測値で埋める（Codex CLI で短く）。x86 未計測を明記。
- 公開バンドル `var/dist/publish/onnx/`（fp32/、int8/、config.json、tokenizer.model、preprocessor.npz、README.md、LICENSE、NOTICE）→ `hf repos create basyauma-technology/Haas-ja-v1.0-onnx --type model --public` → `hf upload`。往復確認（fp32 と int8 の両方で transcribe）。README と台帳を更新。

### Task 14（Mac）: MLX 量子化版（8bit / 4bit）

- 変換: `convert-mlx --nemo ... --out DIR --quantize 8|4 [--group-size 64]`。`mlx.nn.quantize` 相当の量子化を線形層（エンコーダ FF・注意、予測ネットワーク、結合）に適用し、畳み込み・正規化・埋め込みは bf16 のまま。`config.json` に `quantization: {"bits": 8, "group_size": 64, "targets": "linear"}` を記録。safetensors には `weight`/`scales`/`biases` を MLX の規約どおり保存。変換は torch 側で mlx を使わず行うのが難しければ、変換器のこの経路だけ mlx 必須（macOS 限定）にしてよい。
- 読み込み: `HaasMLX.load` が `config.json` の `quantization` を見て `nn.quantize` を適用してから `load_weights(strict=True)`。bf16 版はこれまでどおり。
- 検証: tiny/parity モデルで量子化→読み込みの往復と、bf16 との出力差が bits に応じた許容内（8bit: logits 相対 1e-2 程度、4bit: トークン列一致率で評価）。実モデルで FLEURS・JVNV の CER と RTFx、ピークメモリを bf16 と同じ手順で計測。
- 合格基準: 公表値との差 0.3 ポイント以内の版だけ公開。外れた版は数値を記録し、公開は見送る（カードに「未公開の理由」を一行）。
- 公開: `basyauma-technology/Haas-ja-v1.0-mlx-8bit`、`-4bit`（合格した版のみ）。カードは bf16 版を基に Codex CLI で短く。bf16 版カードと README に量子化版へのリンクを追加。
- 検証記録 `docs/verification/2026-11-08-mac-mlx-quant.md`。

### 文書の方針（全タスク共通）

- 公開文書（カード・README）の本文で「basyauma-technology」に言及しない。リポジトリ ID と URL は残す。
- 要点だけを短く。日本語→英語。数字・英字と日本語の間に半角スペースを入れない。
- 文章の執筆・リライトは Codex CLI（`codex exec`）に任せ、事実・数値・免責・ライセンス表記は変えない。
