"""asr-ja-benchの書き起こしを作り、jacerでCER strict / flex / kanaを測る。

1発話ずつpipeline.transcribeに通す（バッチ1）。longform_threshold_secondsより長い発話は自動で分割認識になり、
その件数をsummaryに残す。CERはサブセット全体をjacer.cer(refs, hyps, jacer.METRICS)で1回計算する
（コーパス単位のマイクロ平均。公式の評価スクリプトと同じ）。

出力（--out DIR）:
    DIR/<subset>/hyps.jsonl     {"id", "reference", "hypothesis", "duration", "seconds"}
    DIR/<subset>/summary.json   CER（%）、発話数、音声合計秒、処理合計秒、RTFx、jacerの版、環境
    DIR/summary.md              全サブセットの表

処理時間は音声読み込みを除いたtranscribeだけの時間。計測前に先頭の発話で1回ウォームアップする（計測に含めない）。
ファイルはすべてUTF-8・改行LFで書く（Windowsでも同じ）。テストではmain(argv, cer_fn=...)でjacerの代わりを渡せる。

実行例:
    uv run --extra eval --extra mlx python scripts/eval_bench.py --model var/dist/mlx --backend mlx \
        --bench var/bench/asr-ja-bench-v1.1 --subset fleurs jvnv --out var/bench/results/mlx-bf16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchlib

CerFn = Callable[[Sequence[str], Sequence[str]], dict[str, float]]


def jacer_scorer() -> tuple[CerFn, dict]:
    """jacerの3指標（0〜1）を返す関数と、summaryに残すjacerの情報。"""
    import jacer

    def cer(refs: Sequence[str], hyps: Sequence[str]) -> dict[str, float]:
        return jacer.cer(list(refs), list(hyps), jacer.METRICS)

    return cer, {
        "scorer": "jacer",
        "jacer_metrics_version": jacer.METRICS_VERSION,
        "jacer_metrics": list(jacer.METRICS),
    }


def evaluate_subset(
    backend, tokenizer, bench: Path, subset: str, out_dir: Path, args, env: dict, cer_fn: CerFn, scorer: dict
) -> dict:
    from haas_ja_local.audio import load_audio
    from haas_ja_local.pipeline import transcribe

    utterances = benchlib.read_manifest(bench, subset, limit=args.limit)
    sample_rate = backend.config.sample_rate
    threshold = args.longform_threshold

    # ウォームアップ（mlxのカーネル準備などを計測から外す）
    transcribe(backend, tokenizer, load_audio(utterances[0].audio, sample_rate=sample_rate))

    refs: list[str] = []
    hyps: list[str] = []
    audio_total = 0.0
    proc_total = 0.0
    n_longform = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "hyps.jsonl").open("w", encoding="utf-8", newline="") as f:
        for i, utt in enumerate(utterances, 1):
            audio = load_audio(utt.audio, sample_rate=sample_rate)
            duration = len(audio) / sample_rate
            if len(audio) > threshold * sample_rate:
                n_longform += 1
            start = time.perf_counter()
            result = transcribe(backend, tokenizer, audio, longform_threshold_seconds=threshold)
            seconds = time.perf_counter() - start
            audio_total += duration
            proc_total += seconds
            refs.append(utt.text)
            hyps.append(result.text)
            row = {
                "id": utt.id,
                "reference": utt.text,
                "hypothesis": result.text,
                "duration": duration,
                "seconds": seconds,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(
                    f"{subset}: {i}/{len(utterances)} ({benchlib.rtfx(audio_total, proc_total):.1f}x)", file=sys.stderr
                )

    cer = benchlib.cer_percent(cer_fn(refs, hyps))
    summary = {
        "subset": subset,
        "cer": cer,
        "published": benchlib.published_delta(subset, cer["cer_strict"]) if args.limit is None else None,
        "n_utterances": len(utterances),
        "n_longform": n_longform,
        "longform_threshold_seconds": threshold,
        "audio_seconds": audio_total,
        "processing_seconds": proc_total,
        "rtfx": benchlib.rtfx(audio_total, proc_total),
        "limit": args.limit,
        **scorer,
        "bench": str(bench),
        "conditions": "batch 1, one utterance at a time, 1 warm-up utterance excluded, audio loading excluded",
        "environment": env,
    }
    benchlib.write_json(out_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None, *, cer_fn: CerFn | None = None) -> int:
    """cer_fnを渡すとjacerの代わりに使う（refs, hypsを受け、cer_strict・cer_flex・cer_kanaを0〜1で返す）。"""
    benchlib.configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    benchlib.add_model_arguments(parser)
    parser.add_argument("--bench", type=Path, required=True, help="rebuilt asr-ja-bench directory")
    parser.add_argument("--subset", nargs="+", required=True, help="subset names (<subset>.jsonl in --bench)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--limit", type=int, help="only the first N utterances of each subset")
    parser.add_argument("--longform-threshold", type=float, default=30.0, help="seconds (default 30)")
    args = parser.parse_args(argv)

    if cer_fn is None:
        cer_fn, scorer = jacer_scorer()
    else:
        scorer = {"scorer": getattr(cer_fn, "__name__", "custom")}
    backend, tokenizer = benchlib.load_backend(args.model, args.backend, variant=args.variant, threads=args.threads)
    env = benchlib.environment(backend, args.backend, args.model, variant=args.variant, threads=args.threads)
    summaries = []
    for subset in args.subset:
        s = evaluate_subset(backend, tokenizer, args.bench, subset, args.out / subset, args, env, cer_fn, scorer)
        summaries.append(s)
        print(
            json.dumps({k: s[k] for k in ("subset", "cer", "n_utterances", "n_longform", "rtfx")}, ensure_ascii=False)
        )
    table = benchlib.markdown_table(summaries)
    (args.out / "summary.md").write_text(table, encoding="utf-8", newline="")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
