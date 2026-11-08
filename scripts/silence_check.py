"""無音と弱いホワイトノイズ（16kHz、10秒）を認識し、文字が出ないかを見る。

信号: デジタル無音（全サンプル0）と、標準偏差1e-3・1e-2・5e-2のガウス白色雑音
（numpy.random.default_rng(0).standard_normal、float32）。

実行例:
    uv run --extra mlx python scripts/silence_check.py --model var/dist/mlx \
        --out var/bench/results/aux/silence-mlx.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
SECONDS = 10.0
NOISE_STDS = (1e-3, 1e-2, 5e-2)


def signals(seed: int = 0) -> dict[str, np.ndarray]:
    n = int(SAMPLE_RATE * SECONDS)
    out = {"silence": np.zeros(n, dtype=np.float32)}
    for std in NOISE_STDS:
        rng = np.random.default_rng(seed)
        out[f"white_noise_std_{std:g}"] = (rng.standard_normal(n) * std).astype(np.float32)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")  # Windowsのcp932に"▁"が無い
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="model directory or .nemo file")
    parser.add_argument("--backend", choices=["mlx", "onnx", "torch"], help="override backend detection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="output JSON file")
    args = parser.parse_args(argv)

    from haas_ja_local.pipeline import load_model, transcribe

    backend, tokenizer = load_model(args.model, backend=args.backend)
    rows = []
    for name, audio in signals(args.seed).items():
        result = transcribe(backend, tokenizer, audio)
        ids = [t.id for t in result.tokens]
        rows.append(
            {
                "signal": name,
                "seconds": len(audio) / SAMPLE_RATE,
                "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
                "text": result.text,
                "token_ids": ids,
                "pieces": tokenizer.pieces(ids),
            }
        )
    report = {"model": str(args.model), "sample_rate": SAMPLE_RATE, "seed": args.seed, "results": rows}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8", newline="")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
