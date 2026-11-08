import json
from pathlib import Path

import pytest

from haas_ja_local.config import HaasConfig, QuantizationConfig

EXCERPT = Path(__file__).parent / "data" / "model_config_excerpt.yaml"


def test_defaults_match_spec():
    c = HaasConfig()
    assert (c.blank_id, c.frame_seconds, c.kda_state_width) == (3072, 0.08, 20480)
    assert c.durations == (0, 1, 2, 3, 4) and c.max_symbols == 10


def test_json_roundtrip(tmp_path):
    c = HaasConfig(source_sha256="ab" * 32)
    c.to_json(tmp_path / "config.json")
    assert HaasConfig.from_json(tmp_path / "config.json") == c


def test_json_has_all_fields_and_durations_as_list(tmp_path):
    HaasConfig().to_json(tmp_path / "config.json")
    data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert data["durations"] == [0, 1, 2, 3, 4]
    # quantizationは量子化版にだけ書く（bf16版・ONNX版のconfig.jsonは従来と同じ）
    assert set(data) == set(HaasConfig.__dataclass_fields__) - {"quantization"}


def test_quantization_json_roundtrip(tmp_path):
    c = HaasConfig(source_sha256="ab" * 32, quantization=QuantizationConfig(bits=4, group_size=64))
    c.to_json(tmp_path / "config.json")
    data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert data["quantization"] == {"bits": 4, "group_size": 64, "targets": "linear"}
    assert HaasConfig.from_json(tmp_path / "config.json") == c
    assert hash(c) == hash(HaasConfig.from_json(tmp_path / "config.json"))


def test_quantization_defaults_to_none_for_existing_config_json(tmp_path):
    # 公開済みのbf16版のconfig.json（quantizationなし）はそのまま読める
    HaasConfig().to_json(tmp_path / "config.json")
    assert HaasConfig.from_json(tmp_path / "config.json").quantization is None


@pytest.mark.parametrize(("bits", "group_size", "targets"), [(3, 64, "linear"), (8, 48, "linear"), (8, 64, "all")])
def test_quantization_rejects_unsupported_values(bits, group_size, targets):
    with pytest.raises(ValueError):
        QuantizationConfig(bits=bits, group_size=group_size, targets=targets)


def test_from_model_config_yaml_reads_upstream_values():
    text = EXCERPT.read_text()
    c = HaasConfig.from_model_config_yaml(text)
    assert (c.d_model, c.n_layers, c.kda_heads, c.kda_head_dim, c.kda_conv_size) == (1024, 24, 5, 64, 4)


def test_from_model_config_yaml_matches_defaults():
    assert HaasConfig.from_model_config_yaml(EXCERPT.read_text()) == HaasConfig()


def test_from_model_config_yaml_reads_non_default_values():
    text = (
        EXCERPT.read_text().replace("kda_num_heads: 5", "kda_num_heads: 3").replace("max_symbols: 10", "max_symbols: 7")
    )
    c = HaasConfig.from_model_config_yaml(text)
    assert (c.kda_heads, c.max_symbols) == (3, 7)


def test_from_model_config_yaml_missing_key_raises():
    text = EXCERPT.read_text().replace("kda_head_dim: 64\n", "")
    with pytest.raises(ValueError, match="kda_head_dim"):
        HaasConfig.from_model_config_yaml(text)
