"""NeMoを使わずに.nemo（tar）からHaas-ja-v1.0の重み・構成・トークナイザを読む。

torchはconvert extraでのみ入るため、関数内でimportする。
"""

from __future__ import annotations

import hashlib
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from haas_ja_local.config import HaasConfig

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class NemoCheckpoint:
    config: HaasConfig
    state_dict: dict[str, torch.Tensor]
    tokenizer_model: bytes
    sha256: str


def expected_shapes(config: HaasConfig) -> dict[str, tuple[int, ...]]:
    """元チェックポイントの各キーの形状（num_batches_tracked含む）。"""
    c = config
    d, ch, p = c.d_model, c.subsampling_channels, c.pred_hidden
    ff = d * c.ff_expansion
    head_k = c.kda_head_dim
    kda_d = c.kda_heads * head_k
    sub_freq = -(-c.n_mels // c.subsampling_factor)
    shapes: dict[str, tuple[int, ...]] = {
        "preprocessor.featurizer.window": (c.win_length,),
        "preprocessor.featurizer.fb": (1, c.n_mels, c.n_fft // 2 + 1),
        "encoder.pre_encode.out.weight": (d, ch * sub_freq),
        "encoder.pre_encode.out.bias": (d,),
    }
    for idx, weight in (
        ("0", (ch, 1, 3, 3)),
        ("2", (ch, 1, 3, 3)),
        ("3", (ch, ch, 1, 1)),
        ("5", (ch, 1, 3, 3)),
        ("6", (ch, ch, 1, 1)),
    ):
        shapes[f"encoder.pre_encode.conv.{idx}.weight"] = weight
        shapes[f"encoder.pre_encode.conv.{idx}.bias"] = (ch,)

    for i in range(c.n_layers):
        prefix = f"encoder.layers.{i}."
        layer: dict[str, tuple[int, ...]] = {
            "feed_forward1.linear1.weight": (ff, d),
            "feed_forward1.linear2.weight": (d, ff),
            "conv.pointwise_conv1.weight": (2 * d, d, 1),
            "conv.depthwise_conv.weight": (d, 1, c.conv_kernel_size),
            "conv.batch_norm.num_batches_tracked": (),
            "conv.pointwise_conv2.weight": (d, d, 1),
            "self_attn.pos_bias_u": (c.n_heads, d // c.n_heads),
            "self_attn.pos_bias_v": (c.n_heads, d // c.n_heads),
            "feed_forward2.linear1.weight": (ff, d),
            "feed_forward2.linear2.weight": (d, ff),
        }
        for norm in ("norm_feed_forward1", "norm_conv", "norm_self_att", "norm_feed_forward2", "norm_out"):
            layer[f"{norm}.weight"] = (d,)
            layer[f"{norm}.bias"] = (d,)
        for stat in ("weight", "bias", "running_mean", "running_var"):
            layer[f"conv.batch_norm.{stat}"] = (d,)
        for proj in ("q", "k", "v", "out", "pos"):
            layer[f"self_attn.linear_{proj}.weight"] = (d, d)
        shapes.update({prefix + k: v for k, v in layer.items()})

    shapes["decoder.prediction.embed.weight"] = (c.vocab_size + 1, p)
    for i in range(c.pred_layers):
        prefix = f"decoder.prediction.dec_rnn.layers.{i}."
        layer = {
            "A_log": (c.kda_heads,),
            "dt_bias": (kda_d,),
            "f_proj.0.weight": (head_k, p),
            "f_proj.1.weight": (kda_d, head_k),
            "b_proj.weight": (c.kda_heads, p),
            "g_proj.0.weight": (head_k, p),
            "g_proj.1.weight": (kda_d, head_k),
            "g_proj.1.bias": (kda_d,),
            "o_norm.weight": (head_k,),
            "o_proj.weight": (p, kda_d),
        }
        for x in ("q", "k", "v"):
            layer[f"{x}_proj.weight"] = (kda_d, p)
            layer[f"{x}_conv1d.weight"] = (kda_d, 1, c.kda_conv_size)
        shapes.update({prefix + k: v for k, v in layer.items()})
        shapes[f"decoder.prediction.dec_rnn.norms.{i}.weight"] = (p,)
        shapes[f"decoder.prediction.dec_rnn.norms.{i}.bias"] = (p,)
    shapes["decoder.prediction.dec_rnn.final_norm.weight"] = (p,)
    shapes["decoder.prediction.dec_rnn.final_norm.bias"] = (p,)

    n_outputs = c.vocab_size + 1 + len(c.durations)
    shapes.update(
        {
            "joint.pred.weight": (c.joint_hidden, p),
            "joint.pred.bias": (c.joint_hidden,),
            "joint.enc.weight": (c.joint_hidden, d),
            "joint.enc.bias": (c.joint_hidden,),
            "joint.joint_net.2.weight": (n_outputs, c.joint_hidden),
            "joint.joint_net.2.bias": (n_outputs,),
        }
    )
    return shapes


def expected_keys(config: HaasConfig) -> frozenset[str]:
    """元チェックポイントのキー名（num_batches_tracked含む）。"""
    return frozenset(expected_shapes(config))


def read_nemo(path: Path) -> NemoCheckpoint:
    """.nemoから構成・state_dict・トークナイザを読む。構成のsource_sha256にはファイル全体のSHA-256を入れる。"""
    import torch

    path = Path(path)
    sha256 = _file_sha256(path)
    with tarfile.open(path, "r:*") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]

        def member(suffix: str) -> tarfile.TarInfo:
            found = [m for m in members if m.name.endswith(suffix)]
            if len(found) != 1:
                raise ValueError(f"{path}: expected one member ending with {suffix!r}, found {[m.name for m in found]}")
            return found[0]

        config_text = _read_member(tar, member("model_config.yaml")).decode("utf-8")
        tokenizer_model = _read_member(tar, member("_tokenizer.model"))
        weights = tar.extractfile(member("model_weights.ckpt"))
        assert weights is not None
        with weights:
            state_dict = torch.load(weights, map_location="cpu", weights_only=True)

    config = replace(HaasConfig.from_model_config_yaml(config_text), source_sha256=sha256)
    return NemoCheckpoint(config=config, state_dict=dict(state_dict), tokenizer_model=tokenizer_model, sha256=sha256)


def _read_member(tar: tarfile.TarFile, info: tarfile.TarInfo) -> bytes:
    f = tar.extractfile(info)
    assert f is not None
    with f:
        return f.read()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()
