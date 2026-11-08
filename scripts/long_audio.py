"""長時間音声の確認用。asr-ja-benchの発話をつないだ音声を作り、CLIの出力を採点する。

make: <subset>.jsonlの先頭から発話を選び、間に無音を挟んでつなぎ、FLACと内訳JSON（id・正解文・開始秒）を書く。
score: `haas-ja-local transcribe --json --timestamps`の出力を読み、次を出す。
    - つないだ正解文（区切りなしで連結）に対するCER strict / flex / kana
    - 同じ発話を1発話ずつ認識した結果（eval_bench.pyのhyps.jsonl）を連結した場合のCER（--hyps）。差が分割境界での損失
    - 分割数（split_at_quiet_pointsを同じ音声に適用）、各断片の境界秒
    - トークン時刻が単調非減少か、最後のトークンの終了秒と音声長

実行例:
    uv run python scripts/long_audio.py make --bench var/bench/asr-ja-bench-v1.1 --subset fleurs --count 150 \
        --out var/bench/long/fleurs_concat_20min.flac
    /usr/bin/time -l uv run haas-ja-local transcribe var/dist/mlx var/bench/long/fleurs_concat_20min.flac \
        --json --timestamps > var/bench/long/out.json
    uv run --extra eval python scripts/long_audio.py score --audio var/bench/long/fleurs_concat_20min.flac \
        --transcript var/bench/long/out.json --hyps var/bench/results/mlx-bf16/fleurs/hyps.jsonl \
        --out var/bench/results/mlx-bf16-long.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchlib

SAMPLE_RATE = 16000


def is_monotonic(starts: list[float]) -> bool:
    return all(a <= b for a, b in itertools.pairwise(starts))


def make(args: argparse.Namespace) -> int:
    import soundfile as sf

    from haas_ja_local.audio import load_audio

    utterances = benchlib.read_manifest(args.bench, args.subset, limit=args.count)
    silence = np.zeros(int(args.silence * SAMPLE_RATE), dtype=np.float32)
    pieces: list[np.ndarray] = []
    parts = []
    position = 0
    for i, utt in enumerate(utterances):
        if i:
            pieces.append(silence)
            position += len(silence)
        audio = load_audio(utt.audio, sample_rate=SAMPLE_RATE)
        parts.append(
            {"id": utt.id, "text": utt.text, "start": position / SAMPLE_RATE, "seconds": len(audio) / SAMPLE_RATE}
        )
        pieces.append(audio)
        position += len(audio)
    audio = np.concatenate(pieces)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio, SAMPLE_RATE, subtype="PCM_16")
    meta = {
        "bench": str(args.bench),
        "subset": args.subset,
        "silence_seconds": args.silence,
        "seconds": len(audio) / SAMPLE_RATE,
        "n_utterances": len(parts),
        "parts": parts,
    }
    benchlib.write_json(args.out.with_suffix(".json"), meta)
    print(f"{args.out}: {meta['seconds']:.1f} s, {len(parts)} utterances")
    return 0


def score(args: argparse.Namespace) -> int:
    import jacer

    from haas_ja_local.audio import load_audio
    from haas_ja_local.longform import split_at_quiet_points

    meta = json.loads(args.audio.with_suffix(".json").read_text(encoding="utf-8"))
    record = json.loads(args.transcript.read_text(encoding="utf-8").splitlines()[0])
    if "tokens" not in record:
        raise SystemExit(f"{args.transcript}: no token times; run transcribe with --json --timestamps")
    reference = "".join(p["text"] for p in meta["parts"])
    audio = load_audio(args.audio, sample_rate=SAMPLE_RATE)
    seconds = len(audio) / SAMPLE_RATE
    spans = split_at_quiet_points(audio, sample_rate=SAMPLE_RATE, chunk_seconds=args.chunk_seconds)
    starts = [t["start"] for t in record["tokens"]]

    result = {
        "audio": str(args.audio),
        "audio_seconds": seconds,
        "n_utterances": meta["n_utterances"],
        "n_chunks": len(spans),
        "chunk_boundaries_seconds": [s / SAMPLE_RATE for s, _ in spans[1:]],
        "n_tokens": len(starts),
        "token_starts_monotonic": is_monotonic(starts),
        "last_token_end_seconds": record["tokens"][-1]["end"] if record["tokens"] else None,
        "reference_chars": len(reference),
        "hypothesis_chars": len(record["text"]),
        "cer_longform": benchlib.cer_percent(jacer.cer([reference], [record["text"]], jacer.METRICS)),
        "jacer_metrics_version": jacer.METRICS_VERSION,
    }
    if args.hyps is not None:
        by_id = {}
        with args.hyps.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                by_id[row["id"]] = row["hypothesis"]
        per_utt = "".join(by_id[p["id"]] for p in meta["parts"])
        result["hypothesis_chars_per_utterance"] = len(per_utt)
        result["cer_per_utterance_concat"] = benchlib.cer_percent(jacer.cer([reference], [per_utt], jacer.METRICS))
        result["hyps"] = str(args.hyps)
    benchlib.write_json(args.out, result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "chunk_boundaries_seconds"}, ensure_ascii=False, indent=2)
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("make", help="concatenate benchmark utterances into one file")
    p.add_argument("--bench", type=Path, required=True)
    p.add_argument("--subset", default="fleurs")
    p.add_argument("--count", type=int, default=150, help="number of utterances from the start (default 150)")
    p.add_argument("--silence", type=float, default=0.5, help="seconds of silence between utterances (default 0.5)")
    p.add_argument("--out", type=Path, required=True, help="output FLAC (a .json with the parts is written beside it)")

    p = commands.add_parser("score", help="score a CLI --json --timestamps transcript of the concatenated file")
    p.add_argument("--audio", type=Path, required=True, help="FLAC written by make (its .json must be beside it)")
    p.add_argument("--transcript", type=Path, required=True, help="output of haas-ja-local transcribe --json")
    p.add_argument("--hyps", type=Path, help="eval_bench.py hyps.jsonl with the same utterances, for comparison")
    p.add_argument("--chunk-seconds", type=float, default=30.0)
    p.add_argument("--out", type=Path, required=True, help="output JSON")

    args = parser.parse_args()
    return make(args) if args.command == "make" else score(args)


if __name__ == "__main__":
    sys.exit(main())
