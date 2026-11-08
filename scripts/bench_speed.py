"""1発話ずつの処理時間からRTFxとピークメモリを測る（バッチ1）。

モデル読み込み後、先頭--warmup発話（既定3）を計測から除き、残りの発話について
1発話ごとのRTFx（音声長÷処理時間）の中央値・平均と、合計のRTFx（合計音声長÷合計処理時間）を出す。
処理時間は音声読み込みを除いたtranscribeだけの時間。メモリはプロセスのピークRSS（`peak_rss_bytes`、bytes）と、
mlxではmx.get_peak_memory()（Metalの割り当てのピーク）。ピークRSSはmacOS・Linuxではgetrusageのru_maxrss、
Windowsではpsutilのpeak_wset（ピークワーキングセット）で測り、どちらを使ったかを`peak_rss_source`に書く。

実行例:
    uv run --extra mlx python scripts/bench_speed.py --model var/dist/mlx --backend mlx \
        --bench var/bench/asr-ja-bench-v1.1 --subset fleurs --out var/bench/results/mlx-bf16-speed.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchlib


def main(argv: Sequence[str] | None = None) -> int:
    benchlib.configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    benchlib.add_model_arguments(parser)
    parser.add_argument("--bench", type=Path, required=True, help="rebuilt asr-ja-bench directory")
    parser.add_argument("--subset", default="fleurs", help="subset name (default fleurs)")
    parser.add_argument("--limit", type=int, help="only the first N utterances (warm-up included)")
    parser.add_argument("--warmup", type=int, default=3, help="leading utterances excluded from timing (default 3)")
    parser.add_argument("--out", type=Path, required=True, help="output JSON file")
    args = parser.parse_args(argv)

    from haas_ja_local.audio import load_audio
    from haas_ja_local.pipeline import transcribe

    utterances = benchlib.read_manifest(args.bench, args.subset, limit=args.limit)
    start = time.perf_counter()
    backend, tokenizer = benchlib.load_backend(args.model, args.backend, variant=args.variant, threads=args.threads)
    load_seconds = time.perf_counter() - start
    memory_after_load = benchlib.peak_memory()

    mx = None
    if args.backend == "mlx":
        import mlx.core as mx

        peak_after_load = mx.get_peak_memory()

    sample_rate = backend.config.sample_rate
    audio_seconds: list[float] = []
    processing_seconds: list[float] = []
    for i, utt in enumerate(utterances, 1):
        audio = load_audio(utt.audio, sample_rate=sample_rate)
        t0 = time.perf_counter()
        transcribe(backend, tokenizer, audio)
        processing_seconds.append(time.perf_counter() - t0)
        audio_seconds.append(len(audio) / sample_rate)
        if i % 100 == 0:
            print(f"{i}/{len(utterances)}", file=sys.stderr)

    result = {
        "subset": args.subset,
        "model_load_seconds": load_seconds,
        **benchlib.speed_stats(audio_seconds, processing_seconds, warmup=args.warmup),
        **benchlib.peak_memory(),
        "peak_rss_bytes_after_load": memory_after_load["peak_rss_bytes"],
        "conditions": "batch 1, one utterance at a time, warm-up excluded, audio loading excluded",
        "per_utterance": [
            {"id": u.id, "audio_seconds": a, "processing_seconds": p}
            for u, a, p in zip(utterances, audio_seconds, processing_seconds, strict=True)
        ],
        "environment": benchlib.environment(
            backend, args.backend, args.model, variant=args.variant, threads=args.threads
        ),
    }
    if mx is not None:
        result["mlx_peak_memory_bytes"] = mx.get_peak_memory()
        result["mlx_peak_memory_bytes_after_load"] = peak_after_load
    benchlib.write_json(args.out, result)
    shown = {k: v for k, v in result.items() if k not in ("environment", "per_utterance")}
    print(json.dumps(shown, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
