""".nemoからMLX用のディレクトリ（model.safetensors・config.json・tokenizer.model）を作る。

bf16版はtorchとsafetensorsだけを使い、mlxはimportしない（Windowsでも変換できる）。
キー名は元のチェックポイントのまま、値もbf16のまま変えない。畳み込みの重みだけを、
mlx.nnの軸順（チャンネルが最後）へ並べ替える。num_batches_trackedは推論に使わないので捨てる。

量子化版（quantization指定時）は、bf16版と同じ重みをMLXのモジュール木に読み込み、線形層だけを
mlx.nn.quantize（affine）で量子化して保存する。この経路だけmlxが必要で、Apple SiliconのmacOSに限る。
量子化した線形層は元のキーの下に`weight`（uint32に詰めた値）・`scales`・`biases`を持つ（MLXの規約）。
それ以外のテンソルはbf16版とビット単位で同じ。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from haas_ja_local.config import HaasConfig, QuantizationConfig
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

# 量子化する線形層の重み（mlx_model.HaasModuleのnn.Linearすべて）。畳み込み・正規化・埋め込みは含めない
QUANTIZED_LINEAR_WEIGHTS: tuple[str, ...] = (
    r"encoder\.pre_encode\.out\.weight",
    r"encoder\.layers\.\d+\.feed_forward[12]\.linear[12]\.weight",
    r"encoder\.layers\.\d+\.self_attn\.linear_(q|k|v|out|pos)\.weight",
    r"decoder\.prediction\.dec_rnn\.layers\.\d+\.([qkvbo]_proj|[fg]_proj\.[01])\.weight",
    r"joint\.(enc|pred|joint_net\.2)\.weight",
)


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


def is_quantized_linear_weight(key: str) -> bool:
    """量子化版で量子化する線形層の重みのキーか。"""
    return any(re.fullmatch(pattern, key) for pattern in QUANTIZED_LINEAR_WEIGHTS)


def expected_quantized_mlx_shapes(config: HaasConfig, quantization: QuantizationConfig) -> dict[str, tuple[int, ...]]:
    """量子化版のmodel.safetensorsの各キーと形状。

    量子化する線形層の`weight` [O,I]を、`weight` [O,I*bits/32]（uint32）と
    `scales`・`biases` [O,I/group_size]に置き換える。
    入力次元Iがgroup_sizeで割り切れない層があればValueError。
    """
    bits, group_size = quantization.bits, quantization.group_size
    out: dict[str, tuple[int, ...]] = {}
    for key, shape in expected_mlx_shapes(config).items():
        if not is_quantized_linear_weight(key):
            out[key] = shape
            continue
        rows, cols = shape
        if cols % group_size:
            raise ValueError(f"{key}: input dimension {cols} is not divisible by group_size {group_size}")
        prefix = key.removesuffix(".weight")
        out[f"{prefix}.weight"] = (rows, cols * bits // 32)
        out[f"{prefix}.scales"] = (rows, cols // group_size)
        out[f"{prefix}.biases"] = (rows, cols // group_size)
    return out


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


def quantize_with_mlx(tensors: dict[str, torch.Tensor], config: HaasConfig, quantization: QuantizationConfig) -> dict:
    """bf16版のテンソルをMLXのモジュール木に読み、線形層を量子化した全パラメータ（キー → mx.array）を返す。

    保存するキー集合と形状がexpected_quantized_mlx_shapesと一致しなければValueError。
    """
    expected = expected_quantized_mlx_shapes(config, quantization)  # 割り切れない層はmlxを読む前に検出する
    if sys.platform != "darwin":
        raise ValueError(f"MLX quantization needs mlx, which runs only on Apple Silicon macOS (this is {sys.platform})")
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from haas_ja_local.mlx_model import HaasModule, quantize_linear_layers

    module = HaasModule(config)
    # bf16 → float32 → bf16は値を変えない（torchのbf16はnumpyへ直接渡せない）
    module.load_weights([(k, mx.array(v.float().numpy()).astype(mx.bfloat16)) for k, v in tensors.items()], strict=True)
    quantize_linear_layers(module, quantization)
    params = dict(tree_flatten(module.parameters()))
    got = {k: tuple(v.shape) for k, v in params.items()}
    if got != expected:
        extra = sorted(set(got) - set(expected))
        missing = sorted(set(expected) - set(got))
        wrong = sorted(k for k in set(got) & set(expected) if got[k] != expected[k])
        raise ValueError(
            f"quantized tensors do not match the expected layout: extra={extra}, missing={missing}, shape={wrong}"
        )
    mx.eval(params)
    return params


def convert_nemo_to_mlx(nemo: Path, out_dir: Path, *, quantization: QuantizationConfig | None = None) -> None:
    """.nemoを読み、out_dirへmodel.safetensors・config.json・tokenizer.modelを書く。

    quantizationを渡すと線形層を量子化した版を書き、config.jsonにquantizationを記録する（mlxが必要）。
    検査はすべて書き込みの前に行い、失敗したときは何も書かない。
    """
    from dataclasses import asdict, replace

    from safetensors.torch import save_file

    from haas_ja_local import __version__

    ckpt = read_nemo(Path(nemo))
    tensors = to_mlx_state_dict(ckpt.state_dict, ckpt.config)
    metadata = {
        "format": "mlx",
        "source_repo": ckpt.config.source_repo,
        "source_sha256": ckpt.sha256,
        "converter": f"haas-ja-local {__version__}",
    }
    config = ckpt.config
    params = None
    if quantization is not None:
        params = quantize_with_mlx(tensors, ckpt.config, quantization)
        metadata["quantization"] = json.dumps(asdict(quantization))
        config = replace(config, quantization=quantization)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "model.safetensors"
    if params is None:
        save_file(tensors, str(weights_path), metadata=metadata)
    else:
        import mlx.core as mx

        mx.save_safetensors(str(weights_path), params, metadata=metadata)
    # safetensorsは一時ファイル経由で書くため0600になる。他のファイルと同じ権限にそろえる
    os.chmod(weights_path, 0o644)
    config.to_json(out_dir / "config.json")
    (out_dir / "tokenizer.model").write_bytes(ckpt.tokenizer_model)
