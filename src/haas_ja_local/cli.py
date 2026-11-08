"""コマンドライン `haas-ja-local`。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from haas_ja_local.audio import load_audio
from haas_ja_local.config import HaasConfig
from haas_ja_local.pipeline import load_model, token_records, transcribe

PROG = "haas-ja-local"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description="NeMo-free local inference for Haas-ja-v1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("transcribe", help="transcribe audio files")
    p.add_argument("model", type=Path, help="converted model directory (MLX / ONNX) or the original .nemo file")
    p.add_argument("audio", type=Path, nargs="+", help="audio files (any format soundfile can read)")
    p.add_argument("--backend", choices=["mlx", "onnx", "torch"], help="override backend detection")
    p.add_argument("--timestamps", action="store_true", help="include per-token times in --json output")
    p.add_argument("--json", action="store_true", help="print one JSON object per file")
    p.add_argument("--threads", type=int, help="CPU threads for the onnx backend (ignored by other backends)")
    p.add_argument(
        "--variant", choices=["fp32", "int8"], help="ONNX model variant (onnx backend only; default fp32 when present)"
    )

    p = commands.add_parser("convert-mlx", help="convert the original .nemo checkpoint to an MLX model directory")
    p.add_argument("--nemo", type=Path, required=True, help="path to Haas-ja-v1.0.nemo")
    p.add_argument("--out", type=Path, required=True, help="output directory")

    p = commands.add_parser("convert-onnx", help="convert the original .nemo checkpoint to an ONNX model directory")
    p.add_argument("--nemo", type=Path, required=True, help="path to Haas-ja-v1.0.nemo")
    p.add_argument("--out", type=Path, required=True, help="output directory (fp32/ and optionally int8/)")
    p.add_argument("--int8", action="store_true", help="also write int8/ (ONNX Runtime dynamic quantization)")
    p.add_argument("--exclude-decoder", action="store_true", help="with --int8: keep decoder_joint.onnx in fp32")
    return parser


def _run_transcribe(args: argparse.Namespace) -> int:
    for path in args.audio:
        if not path.is_file():
            print(f"{PROG}: error: audio file not found: {path}", file=sys.stderr)
            return 1
    try:
        backend, tokenizer = load_model(args.model, backend=args.backend, threads=args.threads, variant=args.variant)
    except (FileNotFoundError, ImportError, ValueError) as e:
        print(f"{PROG}: error: {e}", file=sys.stderr)
        return 1

    config = backend.config
    for path in args.audio:
        result = transcribe(backend, tokenizer, load_audio(path, sample_rate=config.sample_rate))
        if args.json:
            record: dict = {"file": str(path), "text": result.text}
            if args.timestamps:
                record["tokens"] = token_records(result.tokens, result.offsets, tokenizer, config.frame_seconds)
            print(json.dumps(record, ensure_ascii=False))
        elif len(args.audio) > 1:
            print(f"{path}\t{result.text}")
        else:
            print(result.text)
    return 0


def _run_convert_mlx(args: argparse.Namespace) -> int:
    from haas_ja_local.convert.mlx import CONVERTED_FILES, convert_nemo_to_mlx

    if not args.nemo.is_file():
        print(f"{PROG}: error: .nemo file not found: {args.nemo}", file=sys.stderr)
        return 1
    try:
        convert_nemo_to_mlx(args.nemo, args.out)
    except ImportError as e:
        print(
            f"{PROG}: error: conversion needs torch and safetensors: pip install 'haas-ja-local[convert]' ({e})",
            file=sys.stderr,
        )
        return 1
    except ValueError as e:
        print(f"{PROG}: error: {e}", file=sys.stderr)
        return 1
    config = HaasConfig.from_json(args.out / "config.json")
    for name in CONVERTED_FILES:
        print(args.out / name)
    print(f"source_sha256: {config.source_sha256}")
    return 0


def _run_convert_onnx(args: argparse.Namespace) -> int:
    if not args.nemo.is_file():
        print(f"{PROG}: error: .nemo file not found: {args.nemo}", file=sys.stderr)
        return 1
    try:
        from haas_ja_local.convert.onnx import export_onnx, quantize_int8

        export_onnx(args.nemo, args.out)
        if args.int8:
            quantize_int8(args.out / "fp32", args.out / "int8", exclude_decoder=args.exclude_decoder)
    except ImportError as e:
        print(
            f"{PROG}: error: conversion needs torch, onnx and onnxruntime: pip install 'haas-ja-local[convert]' ({e})",
            file=sys.stderr,
        )
        return 1
    except ValueError as e:
        print(f"{PROG}: error: {e}", file=sys.stderr)
        return 1
    config = HaasConfig.from_json(args.out / "config.json")
    for variant in ("fp32", "int8") if args.int8 else ("fp32",):
        for name in sorted(p.name for p in (args.out / variant).iterdir()):
            print(args.out / variant / name)
    for name in ("config.json", "tokenizer.model", "preprocessor.npz"):
        print(args.out / name)
    print(f"source_sha256: {config.source_sha256}")
    return 0


def _utf8_stdout_on_windows() -> None:
    """Windowsの標準出力（既定はcp932など）をUTF-8にする。書けない文字は置き換えて落ちないようにする。"""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_stdout_on_windows()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "convert-onnx" and args.exclude_decoder and not args.int8:
        parser.error("--exclude-decoder requires --int8")
    if args.command == "transcribe":
        return _run_transcribe(args)
    if args.command == "convert-mlx":
        return _run_convert_mlx(args)
    if args.command == "convert-onnx":
        return _run_convert_onnx(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
