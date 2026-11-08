"""自前のtorch実装と、同じチェックポイントのfp32 ONNX書き出しを数値で突き合わせる。

各clipについて次を測る。
(a) 自前log_melと参照haas_ort.logmelの差（有効フレームのmax abs）
(b) 同じ自前melを入れたときのエンコーダ出力の相対誤差 max|a-b| / max|b|
(c) 参照greedyがたどった各ステップ（同じエンコーダ出力・トークン・状態）でのjoint logitsの相対誤差
(d) 自前greedy（自前torch）と参照greedy（参照ONNX）のトークン列の一致
(e) 双方の書き起こしと正解文

実行例:
    uv run --extra convert --extra onnx --with librosa python scripts/verify_numerics.py \
        --nemo var/upstream/Haas-ja-v1.0.nemo --reference-dir var/reference/sanami-onnx \
        --clips var/bench/clips --out var/verify/numerics.json

--onnx-dirを渡すと、参照のfp32/の代わりにそのディレクトリのencoder.onnxとdecoder_joint.onnxを比べる
（入出力名はNeMoの書き出しと同じであること）。melとgreedyの実装は常に--reference-dirのhaas_ort.pyを使う。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

THRESHOLDS = {"mel_max_abs": 1e-3, "encoder_rel": 1e-4, "joint_rel": 1e-4}
DECODER_OUTPUTS = ["outputs", "prednet_lengths", "output_states_1", "output_states_2"]


def load_reference_module(reference_dir: Path) -> ModuleType:
    path = reference_dir / "haas_ort.py"
    spec = importlib.util.spec_from_file_location("haas_ort_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # 参照ディレクトリに__pycache__を作らない
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def make_sessions(onnx_dir: Path) -> tuple[Any, Any]:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    enc = ort.InferenceSession(str(onnx_dir / "encoder.onnx"), so, providers=providers)
    dec = ort.InferenceSession(str(onnx_dir / "decoder_joint.onnx"), so, providers=providers)
    return enc, dec


class RecordingDecoder:
    """decoder_jointのセッションを包み、runの入出力を記録する。出力は名前で並べ直す。"""

    def __init__(self, session: Any) -> None:
        self.session = session
        self.calls: list[tuple[dict[str, np.ndarray], list[np.ndarray]]] = []

    def get_inputs(self):
        return self.session.get_inputs()

    def run(self, _output_names, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        out = self.session.run(DECODER_OUTPUTS, feed)
        self.calls.append(({k: np.array(v, copy=True) for k, v in feed.items()}, out))
        return out


def run_reference_encoder(enc_sess: Any, mel: np.ndarray, n_frames: int) -> tuple[np.ndarray, int]:
    """mel [128,T_pad] → (enc [1,1024,T'], 有効長)。"""
    enc, enc_len = enc_sess.run(
        ["outputs", "encoded_lengths"],
        {"audio_signal": mel[None].astype(np.float32), "length": np.array([n_frames], np.int64)},
    )
    return enc, int(enc_len[0])


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def compare_clip(model, tok, ref: ModuleType, enc_sess: Any, dec_sess: Any, wav: Path) -> dict[str, Any]:
    import torch

    from haas_ja_local.audio import load_audio
    from haas_ja_local.decode import greedy_tdt
    from haas_ja_local.features import log_mel

    cfg = model.config
    samples = load_audio(wav)
    result: dict[str, Any] = {"samples": int(samples.shape[0])}

    # (a) mel
    mel, n = log_mel(samples, model.window_np, model.fb_np, cfg)
    ref_mel, ref_n = ref.logmel(samples)
    ref_mel = ref_mel[0]
    result["mel_frames"] = n
    result["mel_frames_reference"] = int(ref_n)
    result["mel_shape"] = list(mel.shape)
    result["mel_shape_reference"] = list(ref_mel.shape)
    result["mel_max_abs"] = float(np.max(np.abs(mel[:, :n] - ref_mel[:, :n])))
    # 窓とフィルタバンクそのものの差も記録する（参照はlibrosaで作り直してbf16に丸めている）
    ref_window = ref._WINDOW
    result["window_max_abs"] = float(np.max(np.abs(model.window_np - ref_window)))
    result["fb_max_abs"] = float(np.max(np.abs(model.fb_np - ref._FB)))

    # (b) エンコーダ。どちらにも自前のmelを入れる
    ours_enc = model.encode(mel, n).numpy()  # [T',1024]
    ref_enc, ref_len = run_reference_encoder(enc_sess, mel, n)
    ref_enc_t = ref_enc[0, :, :ref_len].T  # [T',1024]
    result["encoded_frames"] = int(ours_enc.shape[0])
    result["encoded_frames_reference"] = ref_len
    t_common = min(ours_enc.shape[0], ref_len)
    scale = float(np.max(np.abs(ref_enc_t[:t_common])))
    per_frame = np.max(np.abs(ours_enc[:t_common] - ref_enc_t[:t_common]), axis=1) / scale
    result["encoder_rel"] = float(per_frame.max())
    result["encoder_rel_last_frame"] = float(per_frame[-1])
    result["encoder_rel_excl_last"] = float(per_frame[:-1].max()) if t_common > 1 else 0.0
    over = np.nonzero(per_frame > THRESHOLDS["encoder_rel"])[0]
    result["encoder_first_frame_over"] = int(over[0]) if over.size else None
    result["encoder_frames_over"] = int(over.size)
    result["encoder_worst_frames"] = [
        {"frame": int(i), "rel": float(per_frame[i])} for i in np.argsort(per_frame)[::-1][:5]
    ]

    # (c) 参照greedyの経路でjoint logitsを比べる。参照のエンコーダ出力（自前mel）を両方へ入れ、
    # 各ステップの入力状態は参照のものを使う
    recorder = RecordingDecoder(dec_sess)
    ref_tokens_ourmel = ref.greedy_tdt(recorder, ref_enc[:, :, :ref_len], ref_len)
    step_rel, state_rel = [], []
    with torch.inference_mode():
        for feed, out in recorder.calls:
            f = torch.from_numpy(feed["encoder_outputs"][:, :, 0])  # [1,1024]
            token = torch.from_numpy(feed["targets"][:, 0].astype(np.int64))
            rec = torch.from_numpy(feed["input_states_1"])
            conv = torch.from_numpy(feed["input_states_2"])
            g, new_rec, new_conv = model.decoder.prediction(token, rec, conv)
            logits = model.joint(f, g)[0].numpy()
            step_rel.append(rel(logits, out[0][0, 0, 0]))
            state_rel.append(max(rel(new_rec.numpy(), out[2]), rel(new_conv.numpy(), out[3])))
    step_rel_np = np.array(step_rel)
    result["joint_steps"] = len(step_rel)
    result["joint_rel"] = float(step_rel_np.max())
    result["joint_rel_median"] = float(np.median(step_rel_np))
    result["state_rel"] = float(max(state_rel))
    result["joint_worst_step"] = int(step_rel_np.argmax())

    # (d)(e) 自前: 自前mel → 自前torch → 自前greedy。参照: 参照logmel → 参照ONNX → 参照greedy
    ours_tokens = [t.id for t in greedy_tdt(model, model.encode(mel, n))]
    ref_enc_full, ref_len_full = run_reference_encoder(enc_sess, ref_mel, int(ref_n))
    ref_tokens = ref.greedy_tdt(dec_sess, ref_enc_full[:, :, :ref_len_full], ref_len_full)
    result["tokens"] = ours_tokens
    result["tokens_reference"] = [int(x) for x in ref_tokens]
    result["tokens_reference_ourmel"] = [int(x) for x in ref_tokens_ourmel]
    result["tokens_equal"] = ours_tokens == result["tokens_reference"]
    result["tokens_equal_ourmel"] = ours_tokens == result["tokens_reference_ourmel"]
    result["text"] = tok.decode(ours_tokens)
    result["text_reference"] = tok.decode(result["tokens_reference"])
    return result


def passes(r: dict[str, Any]) -> dict[str, bool]:
    # フレーム数が違うと(a)(b)は短い方の範囲しか比べないので、一致も合格条件に入れる
    return {
        "mel_frames": r["mel_frames"] == r["mel_frames_reference"],
        "encoded_frames": r["encoded_frames"] == r["encoded_frames_reference"],
        "a": r["mel_max_abs"] < THRESHOLDS["mel_max_abs"],
        "b": r["encoder_rel"] < THRESHOLDS["encoder_rel"],
        "c": r["joint_rel"] < THRESHOLDS["joint_rel"],
        "d": r["tokens_equal"],
    }


def markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| clip | 秒 | (a) mel max abs | (b) enc rel | (b) 最終フレーム除く | (b) 最終フレーム | (c) joint rel | "
        "(c) 状態 rel | ステップ | (d) 一致 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['samples'] / 16000:.1f} | {r['mel_max_abs']:.2e} | {r['encoder_rel']:.2e} | "
            f"{r['encoder_rel_excl_last']:.2e} | {r['encoder_rel_last_frame']:.2e} | {r['joint_rel']:.2e} | "
            f"{r['state_rel']:.2e} | {r['joint_steps']} | {'yes' if r['tokens_equal'] else 'no'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nemo", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True, help="haas_ort.pyとfp32/を含むディレクトリ")
    parser.add_argument("--clips", type=Path, required=True, help="clips.jsonとWAVのあるディレクトリ")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, help="比べるONNXのディレクトリ（既定は<reference-dir>/fp32）")
    args = parser.parse_args(argv)

    import onnxruntime
    import torch

    from haas_ja_local.torch_model import load_torch_model

    ref = load_reference_module(args.reference_dir)
    onnx_dir = args.onnx_dir or args.reference_dir / "fp32"
    enc_sess, dec_sess = make_sessions(onnx_dir)
    model, tok = load_torch_model(args.nemo)

    clips = json.loads((args.clips / "clips.json").read_text(encoding="utf-8"))
    results = []
    for clip in clips:
        started = time.perf_counter()
        r = compare_clip(model, tok, ref, enc_sess, dec_sess, args.clips / clip["file"])
        r = {"id": clip["id"], "file": clip["file"], "transcript": clip["raw"], **r}
        r["pass"] = passes(r)
        r["seconds_elapsed"] = round(time.perf_counter() - started, 1)
        results.append(r)
        print(f"{clip['id']}: {r['pass']} ({r['seconds_elapsed']}s)", file=sys.stderr)

    summary = {
        "mel_max_abs": max(r["mel_max_abs"] for r in results),
        "encoder_rel": max(r["encoder_rel"] for r in results),
        "encoder_rel_excl_last": max(r["encoder_rel_excl_last"] for r in results),
        "joint_rel": max(r["joint_rel"] for r in results),
        "state_rel": max(r["state_rel"] for r in results),
        "tokens_equal": sum(r["tokens_equal"] for r in results),
        "tokens_equal_ourmel": sum(r["tokens_equal_ourmel"] for r in results),
        "frames_equal": sum(r["pass"]["mel_frames"] and r["pass"]["encoded_frames"] for r in results),
        "clips": len(results),
    }
    report = {
        "onnx_dir": str(onnx_dir),
        "reference_dir": str(args.reference_dir),
        "nemo": str(args.nemo),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "onnxruntime": onnxruntime.__version__,
            "numpy": np.__version__,
        },
        "thresholds": THRESHOLDS,
        "summary": summary,
        "clips": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(markdown_table(results))
    print()
    for r in results:
        print(f"- {r['id']} 正解: {r['transcript']}")
        print(f"  自前: {r['text']}")
        print(f"  参照: {r['text_reference']}")
    print()
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if all(all(r["pass"].values()) for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
