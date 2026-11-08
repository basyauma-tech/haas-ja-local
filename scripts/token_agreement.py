"""MLX版（既定のbf16エンコーダ）とtorch（float32）のトークン列・書き起こしの一致率を測る。

1発話ずつ両方のバックエンドでpipeline.transcribeを実行し、トークンid列が完全に一致した発話と、
書き起こしの文字列が一致した発話の割合を出す。設計§6-3の参考値（合否はCERで判断する）。
jacerがあれば、両方の書き起こしのCER strict / flex / kana（%）も出す。

実行例:
    uv run --extra mlx --extra convert --extra eval python scripts/token_agreement.py --model-mlx var/dist/mlx \
        --nemo var/upstream/Haas-ja-v1.0.nemo --bench var/bench/asr-ja-bench-v1.1 --subset fleurs \
        --out var/bench/results/mlx-vs-torch-fleurs.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchlib


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    """2つのid列が最初に食い違う位置。一致すればNone。片方が他方の先頭部分なら短い方の長さ。"""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """発話ごとの{"id", "mlx_ids", "torch_ids", "mlx_text", "torch_text"}から一致率を集計する。"""
    n = len(rows)
    tokens_equal = [r["mlx_ids"] == r["torch_ids"] for r in rows]
    text_equal = [r["mlx_text"] == r["torch_text"] for r in rows]
    return {
        "n_utterances": n,
        "tokens_equal": sum(tokens_equal),
        "tokens_equal_rate": sum(tokens_equal) / n if n else None,
        "text_equal": sum(text_equal),
        "text_equal_rate": sum(text_equal) / n if n else None,
        "tokens_total_mlx": sum(len(r["mlx_ids"]) for r in rows),
        "tokens_total_torch": sum(len(r["torch_ids"]) for r in rows),
        "mismatched_ids": [r["id"] for r, eq in zip(rows, tokens_equal, strict=True) if not eq],
    }


def main(argv: Sequence[str] | None = None) -> int:
    benchlib.configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-mlx", type=Path, required=True, help="converted MLX model directory")
    parser.add_argument("--nemo", type=Path, required=True, help="original .nemo checkpoint (torch reference)")
    parser.add_argument("--bench", type=Path, required=True, help="rebuilt asr-ja-bench directory")
    parser.add_argument("--subset", default="fleurs", help="subset name (default fleurs)")
    parser.add_argument("--limit", type=int, help="only the first N utterances")
    parser.add_argument("--out", type=Path, required=True, help="output JSON file")
    args = parser.parse_args(argv)

    from haas_ja_local.audio import load_audio
    from haas_ja_local.pipeline import load_model, transcribe

    utterances = benchlib.read_manifest(args.bench, args.subset, limit=args.limit)
    mlx_backend, tokenizer = load_model(args.model_mlx, backend="mlx")
    torch_backend, torch_tokenizer = load_model(args.nemo, backend="torch")
    sample_rate = mlx_backend.config.sample_rate

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for i, utt in enumerate(utterances, 1):
        audio = load_audio(utt.audio, sample_rate=sample_rate)
        mlx_result = transcribe(mlx_backend, tokenizer, audio)
        torch_result = transcribe(torch_backend, torch_tokenizer, audio)
        mlx_ids = [t.id for t in mlx_result.tokens]
        torch_ids = [t.id for t in torch_result.tokens]
        rows.append(
            {
                "id": utt.id,
                "reference": utt.text,
                "duration": len(audio) / sample_rate,
                "mlx_ids": mlx_ids,
                "torch_ids": torch_ids,
                "mlx_text": mlx_result.text,
                "torch_text": torch_result.text,
                "first_divergence": first_divergence(mlx_ids, torch_ids),
            }
        )
        if i % 50 == 0:
            print(f"{i}/{len(utterances)} ({time.perf_counter() - started:.0f}s)", file=sys.stderr)

    result: dict[str, Any] = {
        "subset": args.subset,
        "limit": args.limit,
        "bench": str(args.bench),
        "mlx": "encoder bf16, prediction and joint networks float32 (package default)",
        "torch": "float32, CPU",
        **summarize(rows),
        "n_longform": sum(r["duration"] > 30.0 for r in rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    try:
        import jacer
    except ImportError:
        result["cer"] = None
    else:
        refs = [r["reference"] for r in rows]
        result["cer"] = {
            "mlx": benchlib.cer_percent(jacer.cer(refs, [r["mlx_text"] for r in rows], jacer.METRICS)),
            "torch": benchlib.cer_percent(jacer.cer(refs, [r["torch_text"] for r in rows], jacer.METRICS)),
            "jacer_metrics_version": jacer.METRICS_VERSION,
        }
    result["environment"] = benchlib.environment(mlx_backend, "mlx", args.model_mlx, variant=None, threads=None)
    result["per_utterance"] = rows
    benchlib.write_json(args.out, result)
    shown = {k: v for k, v in result.items() if k not in ("environment", "per_utterance")}
    print(json.dumps(shown, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
