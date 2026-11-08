import json
from pathlib import Path

import pytest

from haas_ja_local.config import HaasConfig

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
    assert set(data) == set(HaasConfig.__dataclass_fields__)


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
