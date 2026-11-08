"""MLX量子化版（8bit / 4bit）の変換と読み込み。"""

import io
import json
import re
import shutil
import stat
from dataclasses import replace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import mlx.nn as nn  # noqa: E402
import torch  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from safetensors import safe_open  # noqa: E402

from haas_ja_local.cli import main  # noqa: E402
from haas_ja_local.config import HaasConfig, QuantizationConfig  # noqa: E402
from haas_ja_local.convert.mlx import (  # noqa: E402
    convert_nemo_to_mlx,
    expected_mlx_shapes,
    expected_quantized_mlx_shapes,
    is_quantized_linear_weight,
)
from haas_ja_local.decode import greedy_tdt  # noqa: E402
from haas_ja_local.mlx_model import HaasMLX, HaasModule  # noqa: E402
from haas_ja_local.pipeline import load_model  # noqa: E402
from tests.conftest import TOKENIZER_MEMBER, make_state_dict, model_config_yaml, write_nemo  # noqa: E402

Q8 = QuantizationConfig(bits=8, group_size=64)
Q4 = QuantizationConfig(bits=4, group_size=64)


@pytest.fixture(scope="module")
def quant_config(parity_config):
    # 線形層の入力次元をすべて64の倍数にする（group_size=64で量子化できる最小の構成）。
    # pre_encode.outの入力はチャンネル16×周波数4=64、KDAの値の次元は2×64=128
    return replace(
        parity_config,
        d_model=64,
        subsampling_channels=16,
        pred_hidden=64,
        kda_heads=2,
        kda_head_dim=64,
        joint_hidden=64,
    )


@pytest.fixture(scope="module")
def quant_nemo(quant_config, tiny_tokenizer_model, tmp_path_factory):
    state_dict = make_state_dict(quant_config, seed=5)
    for key, value in state_dict.items():
        # parity_nemoと同じく重みを大きめにし、各ブロックの寄与を残差に埋もれさせない
        if value.is_floating_point() and not key.endswith("running_var") and not key.startswith("preprocessor."):
            state_dict[key] = value * 3
    weights = io.BytesIO()
    torch.save(state_dict, weights)
    members = {
        TOKENIZER_MEMBER: tiny_tokenizer_model,
        "./model_config.yaml": model_config_yaml(quant_config).encode("utf-8"),
        "./model_weights.ckpt": weights.getvalue(),
    }
    return write_nemo(tmp_path_factory.mktemp("quant") / "quant.nemo", members)


@pytest.fixture(scope="module")
def bf16_dir(quant_nemo, tmp_path_factory):
    out = tmp_path_factory.mktemp("quant-bf16")
    convert_nemo_to_mlx(quant_nemo, out)
    return out


@pytest.fixture(scope="module")
def q8_dir(quant_nemo, tmp_path_factory):
    out = tmp_path_factory.mktemp("quant-8bit")
    convert_nemo_to_mlx(quant_nemo, out, quantization=Q8)
    return out


@pytest.fixture(scope="module")
def q4_dir(quant_nemo, tmp_path_factory):
    out = tmp_path_factory.mktemp("quant-4bit")
    convert_nemo_to_mlx(quant_nemo, out, quantization=Q4)
    return out


def _mel(config, n_frames: int, seed: int = 0) -> np.ndarray:
    padded = -(-n_frames // config.pad_to) * config.pad_to
    return np.random.default_rng(seed).standard_normal((config.n_mels, padded)).astype(np.float32)


def _linear_paths(module: nn.Module) -> set[str]:
    return {path for path, m in module.named_modules() if isinstance(m, nn.Linear)}


# ---------------------------------------------------------------------------
# 量子化する層とキー


def test_quantized_linear_weights_are_exactly_the_linear_layers(quant_config):
    # 変換器がmlxなしで使う正規表現と、MLXのモジュール木のnn.Linearが一致する
    expected = {f"{path}.weight" for path in _linear_paths(HaasModule(quant_config))}
    got = {k for k in expected_mlx_shapes(quant_config) if is_quantized_linear_weight(k)}
    assert got == expected
    assert "encoder.layers.0.self_attn.linear_pos.weight" in got
    assert "encoder.pre_encode.out.weight" in got
    assert "decoder.prediction.dec_rnn.layers.1.g_proj.1.weight" in got
    assert "joint.joint_net.2.weight" in got
    assert "decoder.prediction.embed.weight" not in got
    assert "encoder.layers.0.conv.pointwise_conv1.weight" not in got


@pytest.mark.parametrize("q", [Q8, Q4], ids=["8bit", "4bit"])
def test_expected_quantized_shapes(quant_config, q):
    shapes = expected_quantized_mlx_shapes(quant_config, q)
    out, inp = quant_config.d_model, quant_config.d_model
    assert shapes["encoder.layers.0.self_attn.linear_q.weight"] == (out, inp * q.bits // 32)
    assert shapes["encoder.layers.0.self_attn.linear_q.scales"] == (out, inp // q.group_size)
    assert shapes["encoder.layers.0.self_attn.linear_q.biases"] == (out, inp // q.group_size)
    assert shapes["joint.enc.bias"] == (quant_config.joint_hidden,)  # 線形層のバイアスは量子化しない
    assert "encoder.layers.0.conv.depthwise_conv.scales" not in shapes
    bf16 = expected_mlx_shapes(quant_config)
    n_linear = sum(is_quantized_linear_weight(k) for k in bf16)
    assert len(shapes) == len(bf16) + 2 * n_linear


def test_expected_quantized_shapes_rejects_indivisible_input(tiny_config):
    with pytest.raises(ValueError, match=re.escape("group_size 64")):
        expected_quantized_mlx_shapes(tiny_config, Q8)


# ---------------------------------------------------------------------------
# 変換


@pytest.mark.parametrize(("name", "q"), [("q8_dir", Q8), ("q4_dir", Q4)], ids=["8bit", "4bit"])
def test_convert_quantized_writes_expected_files(request, name, q, quant_nemo, quant_config, tiny_tokenizer_model):
    out = request.getfixturevalue(name)
    assert {p.name for p in out.iterdir()} == {"model.safetensors", "config.json", "tokenizer.model"}
    assert stat.S_IMODE((out / "model.safetensors").stat().st_mode) == 0o644
    assert (out / "tokenizer.model").read_bytes() == tiny_tokenizer_model
    data = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert data["quantization"] == {"bits": q.bits, "group_size": 64, "targets": "linear"}
    cfg = HaasConfig.from_json(out / "config.json")
    assert cfg == replace(quant_config, source_sha256=cfg.source_sha256, quantization=q)

    weights = mx.load(str(out / "model.safetensors"))
    assert {k: tuple(v.shape) for k, v in weights.items()} == expected_quantized_mlx_shapes(quant_config, q)
    for key, value in weights.items():
        if key.endswith(".weight") and is_quantized_linear_weight(key):
            assert value.dtype == mx.uint32, key
        else:
            assert value.dtype == mx.bfloat16, key  # scales・biasesと量子化しない部分はbf16のまま

    with safe_open(out / "model.safetensors", framework="np") as f:
        metadata = f.metadata()
    assert metadata["format"] == "mlx"
    assert json.loads(metadata["quantization"]) == {"bits": q.bits, "group_size": 64, "targets": "linear"}
    assert metadata["source_sha256"] == cfg.source_sha256
    assert metadata["source_repo"] == cfg.source_repo
    assert metadata["converter"] == "haas-ja-local 0.1.0"


def test_quantized_conversion_keeps_other_tensors_bit_identical(q4_dir, bf16_dir):
    quantized = mx.load(str(q4_dir / "model.safetensors"))
    bf16 = mx.load(str(bf16_dir / "model.safetensors"))
    kept = [k for k in bf16 if not is_quantized_linear_weight(k)]
    assert kept
    for key in kept:
        assert quantized[key].dtype == bf16[key].dtype == mx.bfloat16
        assert mx.array_equal(quantized[key], bf16[key]).item(), key


def test_quantized_weights_dequantize_close_to_bf16(q8_dir, bf16_dir):
    quantized = mx.load(str(q8_dir / "model.safetensors"))
    bf16 = mx.load(str(bf16_dir / "model.safetensors"))
    key = "joint.joint_net.2"
    w = mx.dequantize(quantized[f"{key}.weight"], quantized[f"{key}.scales"], quantized[f"{key}.biases"], 64, 8)
    ref = bf16[f"{key}.weight"].astype(mx.float32)
    err = mx.abs(w.astype(mx.float32) - ref).max().item() / mx.abs(ref).max().item()
    assert err < 1e-2


def test_convert_quantized_fails_closed_on_indivisible_input(tiny_nemo, tmp_path):
    with pytest.raises(ValueError, match="group_size"):
        convert_nemo_to_mlx(tiny_nemo, tmp_path / "out", quantization=Q8)
    assert not (tmp_path / "out").exists()


def _quantize_all_but_joint_enc(module, quantization):
    nn.quantize(
        module,
        group_size=quantization.group_size,
        bits=quantization.bits,
        class_predicate=lambda path, m: isinstance(m, nn.Linear) and path != "joint.enc",
    )


def _quantize_linear_and_embedding(module, quantization):
    nn.quantize(
        module,
        group_size=quantization.group_size,
        bits=quantization.bits,
        class_predicate=lambda _path, m: isinstance(m, (nn.Linear, nn.Embedding)),
    )


@pytest.mark.parametrize(
    ("quantize", "message"),
    [
        (_quantize_all_but_joint_enc, "joint.enc.scales"),  # 線形層の量子化漏れ
        (_quantize_linear_and_embedding, "decoder.prediction.embed.scales"),  # 線形層以外の量子化
    ],
)
def test_convert_quantized_fails_closed_on_unexpected_layout(quant_nemo, tmp_path, monkeypatch, quantize, message):
    # 変換器がHaasModuleを量子化した結果のキー集合を検査し、想定と違えば何も書かない
    monkeypatch.setattr("haas_ja_local.mlx_model.quantize_linear_layers", quantize)
    with pytest.raises(ValueError, match=re.escape(message)):
        convert_nemo_to_mlx(quant_nemo, tmp_path / "out", quantization=Q8)
    assert not (tmp_path / "out").exists()


def test_convert_quantized_needs_macos(quant_nemo, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    with pytest.raises(ValueError, match="macOS"):
        convert_nemo_to_mlx(quant_nemo, tmp_path / "out", quantization=Q8)
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# 読み込みと出力


@pytest.mark.parametrize(("name", "q"), [("q8_dir", Q8), ("q4_dir", Q4)], ids=["8bit", "4bit"])
def test_load_quantized_roundtrip(request, name, q, quant_config):
    model = HaasMLX.load(request.getfixturevalue(name))
    assert model.config.quantization == q
    module = model.module
    quantized = {p for p, m in module.named_modules() if isinstance(m, nn.QuantizedLinear)}
    assert quantized == _linear_paths(HaasModule(quant_config))
    assert all(
        m.bits == q.bits and m.group_size == q.group_size
        for _, m in module.named_modules()
        if m.__class__ is nn.QuantizedLinear
    )
    params = dict(tree_flatten(module.parameters()))
    assert {k: tuple(v.shape) for k, v in params.items()} == expected_quantized_mlx_shapes(quant_config, q)
    # 精度の方針はbf16版と同じ。エンコーダはcompute_dtype、予測ネットワークとjointはfloat32
    assert params["encoder.layers.0.feed_forward1.linear1.weight"].dtype == mx.uint32
    assert params["encoder.layers.0.feed_forward1.linear1.scales"].dtype == mx.bfloat16
    assert params["encoder.layers.0.norm_out.weight"].dtype == mx.float32
    assert params["decoder.prediction.dec_rnn.layers.0.q_proj.scales"].dtype == mx.float32
    assert params["joint.enc.biases"].dtype == mx.float32
    assert params["joint.enc.bias"].dtype == mx.float32


def test_load_quantized_rejects_missing_scales(q8_dir, tmp_path):
    for name in ("config.json", "tokenizer.model"):
        shutil.copy(q8_dir / name, tmp_path / name)
    weights = mx.load(str(q8_dir / "model.safetensors"))
    del weights["joint.enc.scales"]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    with pytest.raises(ValueError, match=re.escape("joint.enc.scales")):
        HaasMLX.load(tmp_path)


def test_load_bf16_dir_has_no_quantized_layers(bf16_dir):
    model = HaasMLX.load(bf16_dir)
    assert model.config.quantization is None
    assert not any(isinstance(m, nn.QuantizedLinear) for _, m in model.module.named_modules())


@pytest.mark.parametrize("name", ["q8_dir", "q4_dir"])
def test_load_model_dispatches_quantized_directory_to_mlx(request, name):
    backend, _ = load_model(request.getfixturevalue(name))
    assert isinstance(backend, HaasMLX)
    assert backend.config.quantization is not None


def test_quantized_default_dtypes(q8_dir, quant_config):
    model = HaasMLX.load(q8_dir)
    enc = model.encode(_mel(quant_config, 101), 101)
    assert enc.dtype == mx.bfloat16
    logits, (rec, conv) = model.step(enc, 0, quant_config.blank_id, model.init_state())
    assert logits.dtype == np.float32 and np.isfinite(logits).all()
    assert rec.dtype == conv.dtype == mx.float32


def _relative_errors(model, ref, config, seeds=(0, 1, 2)) -> tuple[float, float]:
    """エンコーダ出力とlogits（t=0から3ステップ）の相対誤差の最大値（max|差| / max|基準|）。"""
    enc_err = logit_err = 0.0
    for seed in seeds:
        mel = _mel(config, 101, seed=seed)
        enc, ref_enc = model.encode(mel, 101), ref.encode(mel, 101)
        a, b = np.array(enc.astype(mx.float32)), np.array(ref_enc.astype(mx.float32))
        enc_err = max(enc_err, np.abs(a - b).max() / np.abs(b).max())
        state, ref_state = model.init_state(), ref.init_state()
        for t, token in ((0, config.blank_id), (1, 3), (2, 7)):
            logits, state = model.step(enc, t, token, state)
            ref_logits, ref_state = ref.step(ref_enc, t, token, ref_state)
            logit_err = max(logit_err, np.abs(logits - ref_logits).max() / np.abs(ref_logits).max())
    return float(enc_err), float(logit_err)


def _dequantized_dir(q_dir, tmp_path):
    """量子化した線形層をfloat32へ逆量子化し、quantizationなしのconfig.jsonで書いたディレクトリ。"""
    cfg = HaasConfig.from_json(q_dir / "config.json")
    q = cfg.quantization
    weights = mx.load(str(q_dir / "model.safetensors"))
    out = {}
    for key, value in weights.items():
        prefix = key.removesuffix(".weight")
        if key.endswith(".weight") and is_quantized_linear_weight(key):
            scales, biases = weights[f"{prefix}.scales"], weights[f"{prefix}.biases"]
            out[key] = mx.dequantize(value, scales.astype(mx.float32), biases.astype(mx.float32), q.group_size, q.bits)
        elif not key.endswith((".scales", ".biases")):
            out[key] = value
    mx.save_safetensors(str(tmp_path / "model.safetensors"), out)
    replace(cfg, quantization=None).to_json(tmp_path / "config.json")
    return tmp_path


@pytest.mark.parametrize("name", ["q8_dir", "q4_dir"])
def test_quantized_model_equals_dequantized_float_model(request, name, quant_config, tmp_path):
    # 量子化の誤差を除くと、QuantizedLinearの経路はfloat32の行列積と同じ値を出す（層の置き換え漏れや取り違えがない）
    q_dir = request.getfixturevalue(name)
    model = HaasMLX.load(q_dir, compute_dtype=mx.float32)
    ref = HaasMLX.load(_dequantized_dir(q_dir, tmp_path), compute_dtype=mx.float32)
    enc_err, logit_err = _relative_errors(model, ref, quant_config)
    assert enc_err < 1e-5
    assert logit_err < 1e-5


# 実測（M5 Max、mlx 0.32.4、float32計算、3つの乱数mel）: 8bitはエンコーダ1.26e-2・logits 2.18e-2、
# 4bitはエンコーダ1.90e-1・logits 1.84e-1。重みが乱数のこのモデルは誤差に敏感で、許容は実測の約2倍にした。
# 実モデルでの合否はCERで判断する
@pytest.mark.parametrize(("name", "tol"), [("q8_dir", 5e-2), ("q4_dir", 4e-1)])
def test_quantized_output_close_to_bf16(request, bf16_dir, quant_config, name, tol):
    # 量子化による差だけを見るため、どちらもfloat32で計算する
    model = HaasMLX.load(request.getfixturevalue(name), compute_dtype=mx.float32)
    ref = HaasMLX.load(bf16_dir, compute_dtype=mx.float32)
    enc_err, logit_err = _relative_errors(model, ref, quant_config)
    print(f"{name}: encoder {enc_err:.2e}, logits {logit_err:.2e}")
    assert enc_err < tol
    assert logit_err < tol


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quantized_8bit_greedy_tokens_equal_bf16(q8_dir, bf16_dir, quant_config, seed):
    model = HaasMLX.load(q8_dir, compute_dtype=mx.float32)
    ref = HaasMLX.load(bf16_dir, compute_dtype=mx.float32)
    mel = _mel(quant_config, 203, seed=seed)
    expected = greedy_tdt(ref, ref.encode(mel, 203))
    assert expected  # 空の列どうしの比較で空振りしない
    assert greedy_tdt(model, model.encode(mel, 203)) == expected


# ---------------------------------------------------------------------------
# CLI


def test_cli_convert_mlx_quantize_flags(quant_nemo, tmp_path, capsys):
    out = tmp_path / "q"
    assert (
        main(["convert-mlx", "--nemo", str(quant_nemo), "--out", str(out), "--quantize", "4", "--group-size", "32"])
        == 0
    )
    printed = capsys.readouterr().out
    assert HaasConfig.from_json(out / "config.json").quantization == QuantizationConfig(bits=4, group_size=32)
    for name in ("model.safetensors", "config.json", "tokenizer.model"):
        assert str(out / name) in printed
    assert "quantization: 4bit, group_size 32" in printed
    assert mx.load(str(out / "model.safetensors"))["joint.enc.scales"].shape == (64, 2)


def test_cli_convert_mlx_quantize_defaults_group_size_64(quant_nemo, tmp_path):
    out = tmp_path / "q"
    assert main(["convert-mlx", "--nemo", str(quant_nemo), "--out", str(out), "--quantize", "8"]) == 0
    assert HaasConfig.from_json(out / "config.json").quantization == Q8


@pytest.mark.parametrize(
    "flags", [["--quantize", "3"], ["--quantize", "8", "--group-size", "48"], ["--group-size", "32"]]
)
def test_cli_convert_mlx_rejects_bad_quantize_flags(quant_nemo, tmp_path, flags):
    with pytest.raises(SystemExit) as e:
        main(["convert-mlx", "--nemo", str(quant_nemo), "--out", str(tmp_path / "q"), *flags])
    assert e.value.code == 2
    assert not (tmp_path / "q").exists()


def test_cli_convert_mlx_quantize_reports_indivisible_input(tiny_nemo, tmp_path, capsys):
    assert main(["convert-mlx", "--nemo", str(tiny_nemo), "--out", str(tmp_path / "q"), "--quantize", "8"]) == 1
    assert "group_size" in capsys.readouterr().err
    assert not (tmp_path / "q").exists()
