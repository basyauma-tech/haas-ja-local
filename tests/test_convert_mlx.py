import json
import re

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from haas_ja_local.cli import main
from haas_ja_local.config import HaasConfig
from haas_ja_local.convert.mlx import convert_nemo_to_mlx, mlx_layout
from haas_ja_local.nemo_checkpoint import expected_keys, read_nemo
from tests.conftest import TOKENIZER_MEMBER, make_state_dict, model_config_yaml, write_nemo


def test_convert_tiny_nemo_writes_expected_files(tiny_nemo, tiny_config, tiny_tokenizer_model, tmp_path):
    out = tmp_path / "mlx"
    convert_nemo_to_mlx(tiny_nemo, out)
    assert {p.name for p in out.iterdir()} == {"model.safetensors", "config.json", "tokenizer.model"}
    cfg = HaasConfig.from_json(out / "config.json")
    assert len(cfg.source_sha256) == 64
    assert cfg == read_nemo(tiny_nemo).config
    assert (out / "tokenizer.model").read_bytes() == tiny_tokenizer_model
    with safe_open(out / "model.safetensors", framework="pt") as f:
        metadata = f.metadata()
    assert metadata["format"] == "mlx"
    assert metadata["source_sha256"] == cfg.source_sha256
    assert metadata["source_repo"] == cfg.source_repo
    assert metadata["converter"] == "haas-ja-local 0.1.0"


def test_layout_rules_cover_every_key(tiny_config):
    layouts = {k: mlx_layout(k) for k in expected_keys(tiny_config)}
    assert {k for k, v in layouts.items() if v == "drop"} == {
        f"encoder.layers.{i}.conv.batch_norm.num_batches_tracked" for i in range(tiny_config.n_layers)
    }
    assert {k for k, v in layouts.items() if v == "conv2d"} == {
        f"encoder.pre_encode.conv.{i}.weight" for i in (0, 2, 3, 5, 6)
    }
    conv1d = {k for k, v in layouts.items() if v == "conv1d"}
    assert len(conv1d) == 3 * tiny_config.n_layers + 3 * tiny_config.pred_layers
    assert "encoder.layers.0.conv.depthwise_conv.weight" in conv1d
    assert "decoder.prediction.dec_rnn.layers.1.v_conv1d.weight" in conv1d
    assert "encoder.pre_encode.conv.0.bias" not in conv1d
    assert mlx_layout("joint.enc.weight") == "direct"
    with pytest.raises(KeyError):
        mlx_layout("optimizer_states.0")


def test_converted_tensors_have_mlx_layout_and_same_values(tiny_nemo, tiny_config, tmp_path):
    convert_nemo_to_mlx(tiny_nemo, tmp_path / "mlx")
    converted = load_file(tmp_path / "mlx" / "model.safetensors")
    original = make_state_dict(tiny_config, seed=0)
    assert set(converted) == {k for k in original if not k.endswith("num_batches_tracked")}
    c = tiny_config
    d, ch = c.d_model, c.subsampling_channels
    kda_d = c.kda_heads * c.kda_head_dim
    assert converted["encoder.pre_encode.conv.0.weight"].shape == (ch, 3, 3, 1)  # [O,H,W,I]
    assert converted["encoder.pre_encode.conv.2.weight"].shape == (ch, 3, 3, 1)
    assert converted["encoder.pre_encode.conv.3.weight"].shape == (ch, 1, 1, ch)
    assert converted["encoder.layers.0.conv.pointwise_conv1.weight"].shape == (2 * d, 1, d)  # [O,K,I]
    assert converted["encoder.layers.0.conv.depthwise_conv.weight"].shape == (d, c.conv_kernel_size, 1)  # [D,K,1]
    assert converted["decoder.prediction.dec_rnn.layers.0.q_conv1d.weight"].shape == (kda_d, c.kda_conv_size, 1)
    for key, value in converted.items():
        assert value.dtype == torch.bfloat16, key
        src = original[key]
        layout = mlx_layout(key)
        if layout == "conv2d":
            back = value.permute(0, 3, 1, 2)
        elif layout == "conv1d":
            back = value.permute(0, 2, 1)
        else:
            assert layout == "direct"
            back = value
        assert back.shape == src.shape, key
        assert torch.equal(back, src), key  # bf16のままビット一致


def _write_tiny_nemo(path, config, state_dict, tokenizer_model):
    import io

    weights = io.BytesIO()
    torch.save(state_dict, weights)
    return write_nemo(
        path,
        {
            TOKENIZER_MEMBER: tokenizer_model,
            "./model_config.yaml": model_config_yaml(config).encode("utf-8"),
            "./model_weights.ckpt": weights.getvalue(),
        },
    )


def test_convert_fails_closed_on_unexpected_key(tiny_config, tiny_tokenizer_model, tmp_path):
    state_dict = make_state_dict(tiny_config, seed=0)
    state_dict["encoder.extra.weight"] = torch.zeros(3, dtype=torch.bfloat16)
    nemo = _write_tiny_nemo(tmp_path / "extra.nemo", tiny_config, state_dict, tiny_tokenizer_model)
    with pytest.raises(ValueError, match=re.escape("encoder.extra.weight")):
        convert_nemo_to_mlx(nemo, tmp_path / "out")
    assert not (tmp_path / "out" / "model.safetensors").exists()


def test_convert_fails_closed_on_missing_key(tiny_config, tiny_tokenizer_model, tmp_path):
    state_dict = make_state_dict(tiny_config, seed=0)
    del state_dict["joint.enc.bias"]
    nemo = _write_tiny_nemo(tmp_path / "missing.nemo", tiny_config, state_dict, tiny_tokenizer_model)
    with pytest.raises(ValueError, match=re.escape("joint.enc.bias")):
        convert_nemo_to_mlx(nemo, tmp_path / "out")


def test_convert_fails_closed_on_wrong_shape(tiny_config, tiny_tokenizer_model, tmp_path):
    state_dict = make_state_dict(tiny_config, seed=0)
    state_dict["joint.enc.weight"] = state_dict["joint.enc.weight"].T.contiguous()
    nemo = _write_tiny_nemo(tmp_path / "shape.nemo", tiny_config, state_dict, tiny_tokenizer_model)
    with pytest.raises(ValueError, match=re.escape("joint.enc.weight")):
        convert_nemo_to_mlx(nemo, tmp_path / "out")


def test_cli_convert_mlx(tiny_nemo, tmp_path, capsys):
    out = tmp_path / "cli-out"
    assert main(["convert-mlx", "--nemo", str(tiny_nemo), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    sha256 = json.loads((out / "config.json").read_text(encoding="utf-8"))["source_sha256"]
    assert sha256 in printed
    for name in ("model.safetensors", "config.json", "tokenizer.model"):
        assert str(out / name) in printed


def test_cli_convert_mlx_missing_nemo(tmp_path, capsys):
    assert main(["convert-mlx", "--nemo", str(tmp_path / "nope.nemo"), "--out", str(tmp_path / "o")]) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_convert_mlx_reports_mismatched_checkpoint(tiny_config, tiny_tokenizer_model, tmp_path, capsys):
    state_dict = make_state_dict(tiny_config, seed=0)
    del state_dict["joint.enc.bias"]
    nemo = _write_tiny_nemo(tmp_path / "missing.nemo", tiny_config, state_dict, tiny_tokenizer_model)
    assert main(["convert-mlx", "--nemo", str(nemo), "--out", str(tmp_path / "o")]) == 1
    assert "joint.enc.bias" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_cli_convert_mlx_reports_missing_torch(tiny_nemo, tmp_path, capsys, monkeypatch):
    import haas_ja_local.convert.mlx as convert_mlx

    def fail(*args, **kwargs):
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(convert_mlx, "convert_nemo_to_mlx", fail)
    assert main(["convert-mlx", "--nemo", str(tiny_nemo), "--out", str(tmp_path / "o")]) == 1
    assert "haas-ja-local[convert]" in capsys.readouterr().err
