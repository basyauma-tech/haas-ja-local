"""評価・速度計測スクリプトの共通部分。

eval_bench.py・bench_speed.py・long_audio.pyから読む。純粋な関数（集計・RTFx・表）はtests/test_benchlib.pyで確かめる。
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 公表値（asr-ja-bench v1.1、CER strict %）。モデルカードと公開記事で小数第2位が違うので幅で持つ
PUBLISHED_CER_STRICT = {"fleurs": (3.65, 3.67), "jvnv": (4.50, 4.53)}
PASS_MARGIN_POINTS = 0.3
METRIC_KEYS = ("cer_strict", "cer_flex", "cer_kana")
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Utterance:
    id: str
    audio: Path
    text: str
    duration: float


def read_manifest(bench_dir: Path, subset: str, *, limit: int | None = None) -> list[Utterance]:
    """<bench_dir>/<subset>.jsonlを読む。音声のパスはbench_dirからの相対（絶対パスならそのまま）。

    音声のキーは`audio`か`audio_filepath`。durationが無い行は0とする（呼び出し側で実長を使う）。
    """
    path = bench_dir / f"{subset}.jsonl"
    items: list[Utterance] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            audio = Path(row.get("audio") or row["audio_filepath"])
            items.append(
                Utterance(
                    id=str(row.get("id", audio.stem)),
                    audio=audio if audio.is_absolute() else bench_dir / audio,
                    text=row["text"],
                    duration=float(row.get("duration", 0.0)),
                )
            )
            if limit is not None and len(items) >= limit:
                break
    return items


def rtfx(audio_seconds: float, processing_seconds: float) -> float:
    """実時間倍率（音声長÷処理時間）。処理時間が0以下ならinf。"""
    if processing_seconds <= 0:
        return float("inf")
    return audio_seconds / processing_seconds


def speed_stats(audio_seconds: Sequence[float], processing_seconds: Sequence[float], *, warmup: int) -> dict[str, Any]:
    """1発話ずつの音声長と処理時間から、先頭warmup発話を除いて速度を集計する。"""
    if len(audio_seconds) != len(processing_seconds):
        raise ValueError("audio_seconds and processing_seconds differ in length")
    audio = list(audio_seconds[warmup:])
    proc = list(processing_seconds[warmup:])
    if not audio:
        raise ValueError(f"no utterances left after excluding {warmup} warm-up utterances")
    per_utt = [rtfx(a, p) for a, p in zip(audio, proc, strict=True)]
    return {
        "n_utterances": len(audio),
        "n_warmup_excluded": warmup,
        "audio_seconds": sum(audio),
        "processing_seconds": sum(proc),
        "rtfx_total": rtfx(sum(audio), sum(proc)),
        "rtfx_median": statistics.median(per_utt),
        "rtfx_mean": statistics.fmean(per_utt),
        "seconds_per_utterance_median": statistics.median(proc),
    }


def peak_memory() -> dict[str, Any]:
    """このプロセスのピークメモリ（bytes）と、その測り方。

    macOS・Linuxはgetrusageのru_maxrss（macOSはbytes、Linuxはkibibytes）。
    Windowsにはresourceが無いので、psutilのピークワーキングセット（peak_wset）を使う。
    """
    if sys.platform != "win32":
        import resource  # Unixだけにある

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
            "peak_rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        }
    import psutil

    return {
        "peak_rss_bytes": int(psutil.Process().memory_info().peak_wset),
        "peak_rss_source": "psutil.Process().memory_info().peak_wset",
    }


def configure_stdio() -> None:
    """標準出力・標準エラーで書けない文字を\\uXXXXにして書く（Windowsのcp932で落ちないように）。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")


def cer_percent(scores: dict[str, float]) -> dict[str, float]:
    """jacer.cerの戻り値（0〜1）を%にする。"""
    return {k: scores[k] * 100.0 for k in METRIC_KEYS}


def published_delta(subset: str, cer_strict_percent: float) -> dict[str, Any] | None:
    """公表値（幅）との差。幅の内側なら0、外側なら近い端との差（符号付き、ポイント）。"""
    if subset not in PUBLISHED_CER_STRICT:
        return None
    lo, hi = PUBLISHED_CER_STRICT[subset]
    if cer_strict_percent < lo:
        delta = cer_strict_percent - lo
    elif cer_strict_percent > hi:
        delta = cer_strict_percent - hi
    else:
        delta = 0.0
    return {
        "published_cer_strict": [lo, hi],
        "delta_points": delta,
        # 浮動小数の誤差で境界ちょうど（0.30000000000000004など）を不合格にしない
        "pass": round(abs(delta), 9) <= PASS_MARGIN_POINTS,
        "margin_points": PASS_MARGIN_POINTS,
    }


def markdown_table(summaries: Iterable[dict[str, Any]]) -> str:
    """subsetごとのsummaryからCER・RTFxのMarkdown表を作る。"""
    lines = [
        "| subset | 発話数 | 音声(時間) | CER strict | CER flex | CER kana | 公表値strict | 差(ポイント) | RTFx |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        cer = s["cer"]
        pub = s.get("published")
        pub_text = f"{pub['published_cer_strict'][0]:.2f}〜{pub['published_cer_strict'][1]:.2f}" if pub else "-"
        delta_text = f"{pub['delta_points']:+.2f}" if pub else "-"
        lines.append(
            f"| {s['subset']} | {s['n_utterances']} | {s['audio_seconds'] / 3600:.2f} | {cer['cer_strict']:.2f} | "
            f"{cer['cer_flex']:.2f} | {cer['cer_kana']:.2f} | {pub_text} | {delta_text} | {s['rtfx']:.1f} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# モデルの読み込みと環境（実機依存。テストしない）


def add_model_arguments(parser) -> None:
    parser.add_argument("--model", type=Path, required=True, help="model directory (mlx/onnx) or .nemo file (torch)")
    parser.add_argument("--backend", choices=["mlx", "onnx", "torch"], required=True)
    parser.add_argument("--variant", choices=["fp32", "int8"], help="ONNX variant (onnx backend only)")
    parser.add_argument("--threads", type=int, help="CPU threads (onnx backend only)")


def load_backend(model: Path, backend: str, *, variant: str | None, threads: int | None):
    """(バックエンド, トークナイザ)を返す。--variantをonnx以外で指定したら終了する。"""
    from haas_ja_local.pipeline import load_model

    if variant is not None and backend != "onnx":
        raise SystemExit("--variant is only for --backend onnx")
    return load_model(model, backend=backend, threads=threads, variant=variant)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def compute_dtype(backend_impl: Any, backend: str) -> str:
    if backend == "mlx":
        return f"encoder {backend_impl.compute_dtype}, decoder/joint mlx.core.float32"
    if backend == "torch":
        return f"torch {backend_impl.joint.enc.weight.dtype}"
    return str(getattr(backend_impl, "variant", "onnx"))


def _memory_total_bytes() -> int | None:
    if sys.platform == "darwin":
        return int(_command(["sysctl", "-n", "hw.memsize"]) or 0) or None
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().total)


def system_info() -> dict[str, Any]:
    """機種・CPU・メモリ・OS・Python。sysctlはmacOSでだけ呼ぶ。"""
    if sys.platform == "darwin":
        chip = _command(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
        os_name = f"macOS {platform.mac_ver()[0]}"
    else:
        chip = platform.processor() or platform.machine()
        os_name = platform.platform()
    return {
        "machine": platform.machine(),
        "chip": chip,
        "memory_bytes": _memory_total_bytes(),
        "os": os_name,
        "python": platform.python_version(),
    }


def environment(backend_impl: Any, backend: str, model: Path, *, variant: str | None, threads: int | None) -> dict:
    import os

    import haas_ja_local

    git_head = _command(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    git_dirty = _command(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"])
    env: dict[str, Any] = {
        **system_info(),
        "haas_ja_local": haas_ja_local.__version__,
        "git_commit": git_head,
        "git_dirty": bool(git_dirty),
        "versions": {
            name: _package_version(name) for name in ("mlx", "onnxruntime", "torch", "numpy", "jacer", "psutil")
        },
        "model": str(model),
        "source_sha256": getattr(backend_impl.config, "source_sha256", "") or None,
        "backend": backend,
        "variant": variant,
        "threads": threads,
        "compute_dtype": compute_dtype(backend_impl, backend),
    }
    if backend == "mlx":
        env["mlx_enable_tf32"] = os.environ.get("MLX_ENABLE_TF32")
        env["mlx_device"] = str(importlib.import_module("mlx.core").default_device())
    return env


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")
