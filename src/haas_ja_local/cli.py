"""コマンドライン `haas-ja-local`。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from haas_ja_local.audio import load_audio
from haas_ja_local.tokenizer import Tokenizer
from haas_ja_local.transcribe import Transcript, load_model, transcribe

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
    return parser


def _token_records(transcript: Transcript, tokenizer: Tokenizer, frame_seconds: float) -> list[dict]:
    records = []
    for token, offset in zip(transcript.tokens, transcript.offsets, strict=True):
        start = offset + token.frame * frame_seconds
        end = start + max(token.duration, 1) * frame_seconds
        text = tokenizer.pieces([token.id])[0].replace("▁", " ")
        records.append({"text": text, "start": round(start, 3), "end": round(end, 3)})
    return records


def _run_transcribe(args: argparse.Namespace) -> int:
    for path in args.audio:
        if not path.is_file():
            print(f"{PROG}: error: audio file not found: {path}", file=sys.stderr)
            return 1
    try:
        backend, tokenizer = load_model(args.model, backend=args.backend, threads=args.threads)
    except (FileNotFoundError, ImportError) as e:
        print(f"{PROG}: error: {e}", file=sys.stderr)
        return 1

    config = backend.config
    for path in args.audio:
        result = transcribe(backend, tokenizer, load_audio(path, sample_rate=config.sample_rate))
        if args.json:
            record: dict = {"file": str(path), "text": result.text}
            if args.timestamps:
                record["tokens"] = _token_records(result, tokenizer, config.frame_seconds)
            print(json.dumps(record, ensure_ascii=False))
        elif len(args.audio) > 1:
            print(f"{path}\t{result.text}")
        else:
            print(result.text)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "transcribe":
        return _run_transcribe(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
