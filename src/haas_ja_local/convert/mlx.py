""".nemoからMLX用のディレクトリ（model.safetensors・config.json・tokenizer.model）を作る。

torchとsafetensorsだけを使い、mlxはimportしない（Windowsでも変換できる）。
キー名は元のチェックポイントのまま、値もbf16のまま変えない。畳み込みの重みだけを、
mlx.nnの軸順（チャンネルが最後）へ並べ替える。num_batches_trackedは推論に使わないので捨てる。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from haas_ja_local.config import HaasConfig
from haas_ja_local.nemo_checkpoint import expected_shapes, read_nemo

if TYPE_CHECKING:
    import torch

# (キーのパターン, 変換)。上から順に最初に一致したものを使う
MLX_LAYOUT: tuple[tuple[str, str], ...] = (
    (r".*\.num_batches_tracked", "drop"),
    # torch [O,I,H,W] → mlx [O,H,W,I]
    (r"encoder\.pre_encode\.conv\.\d+\.weight", "conv2d"),
    # torch [O,I,K] → mlx [O,K,I]。depthwise（[D,1,K] → [D,K,1]）も同じ並べ替え
    (r"encoder\.layers\.\d+\.conv\.(pointwise_conv1|pointwise_conv2|depthwise_conv)\.weight", "conv1d"),
    (r"decoder\.prediction\.dec_rnn\.layers\.\d+\.[qkv]_conv1d\.weight", "conv1d"),
    (r"(preprocessor|encoder|decoder|joint)\.[A-Za-z0-9_.]+", "direct"),
)

_PERMUTATIONS = {"conv2d": (0, 2, 3, 1), "conv1d": (0, 2, 1)}

CONVERTED_FILES = ("model.safetensors", "config.json", "tokenizer.model")


def mlx_layout(key: str) -> str:
    """キーに適用する変換名（"drop"・"conv2d"・"conv1d"・"direct"）。どれにも当たらなければKeyError。"""
    for pattern, layout in MLX_LAYOUT:
        if re.fullmatch(pattern, key):
            return layout
    raise KeyError(f"no MLX layout rule for {key!r}")


def mlx_shape(key: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    """元の形状から変換後の形状を求める。"""
    perm = _PERMUTATIONS.get(mlx_layout(key))
    if perm is None:
        return tuple(shape)
    if len(shape) != len(perm):
        raise ValueError(f"{key}: expected a {len(perm)}-d convolution weight, got shape {tuple(shape)}")
    return tuple(shape[i] for i in perm)


def expected_mlx_shapes(config: HaasConfig) -> dict[str, tuple[int, ...]]:
    """変換後のmodel.safetensorsの各キーと形状（num_batches_trackedを除く）。"""
    return {k: mlx_shape(k, v) for k, v in expected_shapes(config).items() if mlx_layout(k) != "drop"}


def to_mlx_state_dict(state_dict: dict[str, torch.Tensor], config: HaasConfig) -> dict[str, torch.Tensor]:
    """キー集合と形状を検査してから軸を並べ替える。余分・不足・形状違いはValueError。"""
    shapes = expected_shapes(config)
    extra = sorted(set(state_dict) - set(shapes))
    missing = sorted(set(shapes) - set(state_dict))
    if extra or missing:
        raise ValueError(f"checkpoint keys do not match Haas-ja-v1.0: extra={extra}, missing={missing}")
    wrong = sorted(k for k, v in state_dict.items() if tuple(v.shape) != shapes[k])
    if wrong:
        detail = ", ".join(f"{k}: {tuple(state_dict[k].shape)} != {shapes[k]}" for k in wrong)
        raise ValueError(f"checkpoint shapes do not match Haas-ja-v1.0: {detail}")

    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        layout = mlx_layout(key)
        if layout == "drop":
            continue
        if not value.is_floating_point():
            raise ValueError(f"{key}: expected a floating point tensor, got {value.dtype}")
        perm = _PERMUTATIONS.get(layout)
        out[key] = (value.permute(*perm) if perm else value).contiguous()

    expected = expected_mlx_shapes(config)
    if {k: tuple(v.shape) for k, v in out.items()} != expected:
        raise ValueError("converted tensors do not match the expected MLX layout")
    return out


def convert_nemo_to_mlx(nemo: Path, out_dir: Path) -> None:
    """.nemoを読み、out_dirへmodel.safetensors・config.json・tokenizer.modelを書く。

    検査はすべて書き込みの前に行い、失敗したときは何も書かない。
    """
    from safetensors.torch import save_file

    from haas_ja_local import __version__

    ckpt = read_nemo(Path(nemo))
    tensors = to_mlx_state_dict(ckpt.state_dict, ckpt.config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "mlx",
        "source_repo": ckpt.config.source_repo,
        "source_sha256": ckpt.sha256,
        "converter": f"haas-ja-local {__version__}",
    }
    weights_path = out_dir / "model.safetensors"
    save_file(tensors, str(weights_path), metadata=metadata)
    # safetensorsは一時ファイル経由で書くため0600になる。他のファイルと同じ権限にそろえる
    os.chmod(weights_path, 0o644)
    ckpt.config.to_json(out_dir / "config.json")
    (out_dir / "tokenizer.model").write_bytes(ckpt.tokenizer_model)
