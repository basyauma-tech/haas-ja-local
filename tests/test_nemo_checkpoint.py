import hashlib
from dataclasses import replace

import pytest
import torch

from haas_ja_local.config import HaasConfig
from haas_ja_local.nemo_checkpoint import expected_keys, expected_shapes, read_nemo
from tests.conftest import model_config_yaml, write_nemo


def test_expected_keys_count_for_upstream_config():
    assert len(expected_keys(HaasConfig())) == 755


def test_expected_keys_scale_with_layers():
    c = HaasConfig()
    assert len(expected_keys(replace(c, n_layers=1, pred_layers=1))) == 2 + 12 + 29 + 16 + 1 + 2 + 2 + 6


def test_expected_shapes_for_upstream_config():
    shapes = expected_shapes(HaasConfig())
    assert shapes["encoder.pre_encode.out.weight"] == (1024, 4096)
    assert shapes["preprocessor.featurizer.fb"] == (1, 128, 257)
    assert shapes["joint.joint_net.2.weight"] == (3078, 640)
    assert shapes["encoder.layers.0.conv.batch_norm.num_batches_tracked"] == ()


def test_read_tiny_nemo(tiny_nemo, tiny_config, tiny_tokenizer_model):
    ck = read_nemo(tiny_nemo)
    assert ck.sha256 == hashlib.sha256(tiny_nemo.read_bytes()).hexdigest()
    assert ck.config == replace(tiny_config, source_sha256=ck.sha256)
    assert set(ck.state_dict) == expected_keys(tiny_config)
    assert ck.tokenizer_model == tiny_tokenizer_model


def test_tiny_state_dict_matches_shapes_and_dtypes(tiny_state_dict, tiny_config):
    shapes = expected_shapes(tiny_config)
    assert {k: tuple(v.shape) for k, v in tiny_state_dict.items()} == shapes
    for name, tensor in tiny_state_dict.items():
        expected = torch.int64 if name.endswith("num_batches_tracked") else torch.bfloat16
        assert tensor.dtype == expected, name
        if name.endswith("running_var"):
            assert bool((tensor > 0).all()), name


def test_read_nemo_without_tokenizer_raises(tmp_path, tiny_config):
    path = write_nemo(tmp_path / "broken.nemo", {"./model_config.yaml": model_config_yaml(tiny_config).encode()})
    with pytest.raises(ValueError, match=r"_tokenizer\.model"):
        read_nemo(path)


@pytest.mark.real_model
def test_read_real_nemo(real_nemo):
    ck = read_nemo(real_nemo)
    assert ck.sha256 == "c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa"
    assert set(ck.state_dict) == expected_keys(ck.config)
    assert sum(v.numel() for v in ck.state_dict.values()) == 615871544
    # 全体から24層分のnum_batches_tracked（int64、各1要素）を除いたものがbf16の重み
    assert sum(v.numel() for v in ck.state_dict.values() if v.dtype == torch.bfloat16) == 615871544 - 24
    assert ck.config == HaasConfig(source_sha256=ck.sha256)
    assert {k: tuple(v.shape) for k, v in ck.state_dict.items()} == expected_shapes(ck.config)
