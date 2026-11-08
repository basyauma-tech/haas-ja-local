"""scripts/の全モジュールがどのOSでもimportでき、メモリと環境の記録がOSごとの方法で取れること。

CI（macOS・Linux・Windows）で回す。Windowsにはresourceもsysctlも無い。
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchlib  # noqa: E402

SCRIPT_MODULES = sorted(p.stem for p in SCRIPTS.glob("*.py"))


def test_scripts_directory_is_scanned():
    assert {"bench_speed", "benchlib", "eval_bench", "rebuild_bench", "token_agreement", "verify_numerics"} <= set(
        SCRIPT_MODULES
    )


@pytest.mark.parametrize("name", SCRIPT_MODULES)
def test_script_imports_without_resource(monkeypatch, name):
    # Windowsを模してresourceをimportできなくし、import時に読まないことを確かめる
    monkeypatch.setitem(sys.modules, "resource", None)
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    if name != "benchlib":
        assert callable(module.main)


def test_peak_memory_of_this_process():
    memory = benchlib.peak_memory()
    assert set(memory) == {"peak_rss_bytes", "peak_rss_source"}
    assert isinstance(memory["peak_rss_bytes"], int)
    assert 10 * 2**20 < memory["peak_rss_bytes"] < 1 << 40  # Pythonとnumpyを読んだプロセスは10MiBより大きい
    expected = "psutil" if sys.platform == "win32" else "getrusage"
    assert expected in memory["peak_rss_source"]


@pytest.mark.parametrize(("platform", "factor"), [("darwin", 1), ("linux", 1024)])
def test_peak_memory_units_on_unix(monkeypatch, platform, factor):
    fake = types.SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda who: types.SimpleNamespace(ru_maxrss=1000))
    monkeypatch.setitem(sys.modules, "resource", fake)
    monkeypatch.setattr(sys, "platform", platform)
    assert benchlib.peak_memory() == {
        "peak_rss_bytes": 1000 * factor,
        "peak_rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
    }


def test_peak_memory_on_windows_uses_psutil_peak_working_set(monkeypatch):
    info = types.SimpleNamespace(peak_wset=12345)
    fake = types.SimpleNamespace(Process=lambda: types.SimpleNamespace(memory_info=lambda: info))
    monkeypatch.setitem(sys.modules, "psutil", fake)
    monkeypatch.setitem(sys.modules, "resource", None)  # Windowsにはresourceが無い
    monkeypatch.setattr(sys, "platform", "win32")
    assert benchlib.peak_memory() == {
        "peak_rss_bytes": 12345,
        "peak_rss_source": "psutil.Process().memory_info().peak_wset",
    }


def test_system_info_of_this_machine():
    info = benchlib.system_info()
    assert {"machine", "chip", "memory_bytes", "os", "python"} <= set(info)
    assert info["machine"]
    assert info["python"] == f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_system_info_does_not_call_sysctl_off_macos(monkeypatch, platform):
    calls = []
    monkeypatch.setattr(benchlib, "_command", lambda args: calls.append(args))
    fake_psutil = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(total=8 * 2**30))
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(benchlib.platform, "platform", lambda: "Windows-11-10.0.26100-SP0")
    monkeypatch.setattr(sys, "platform", platform)
    info = benchlib.system_info()
    assert not [c for c in calls if c and c[0] == "sysctl"]
    assert info["os"] == "Windows-11-10.0.26100-SP0"
    assert info["memory_bytes"] == 8 * 2**30


def test_environment_with_tiny_torch_backend(tiny_nemo):
    from haas_ja_local.pipeline import load_model

    backend, _ = load_model(tiny_nemo)
    env = benchlib.environment(backend, "torch", tiny_nemo, variant=None, threads=None)
    assert env["backend"] == "torch"
    assert env["compute_dtype"].startswith("torch ")
    assert env["model"] == str(tiny_nemo)
    assert env["python"] and env["machine"]


def test_configure_stdio_never_raises_on_cp932(monkeypatch):
    # Windowsでリダイレクトした標準出力はcp932。cp932に無い文字（𠮷、▁）でも落ちずに書ける
    import io

    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp932")
    monkeypatch.setattr(sys, "stdout", stream)
    benchlib.configure_stdio()
    print("𠮷▁今日")
    stream.flush()
    assert stream.buffer.getvalue().decode("cp932") == "\\U00020bb7\\u2581今日" + "\n"


def _write_tiny_bench(root: Path) -> Path:
    import json

    import numpy as np
    import soundfile as sf

    bench = root / "bench"
    (bench / "audio" / "tiny").mkdir(parents=True)
    rows = []
    for i, text in enumerate(["𠮷野家で今日は", "良い天気ですね"]):
        audio = np.random.default_rng(i).standard_normal(16000 + 4000 * i).astype(np.float32) * 0.1
        sf.write(bench / "audio" / "tiny" / f"u{i}.wav", audio, 16000)
        rows.append({"id": f"u{i}", "audio": f"audio/tiny/u{i}.wav", "text": text, "duration": len(audio) / 16000})
    with (bench / "tiny.jsonl").open("w", encoding="utf-8", newline="") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return bench


def _eval_argv(tiny_nemo, bench, out):
    return [
        "--model",
        str(tiny_nemo),
        "--backend",
        "torch",
        "--bench",
        str(bench),
        "--subset",
        "tiny",
        "--out",
        str(out),
    ]


def test_eval_bench_end_to_end_with_stand_in_cer(tiny_nemo, tmp_path, capsys):
    import json

    import eval_bench

    bench = _write_tiny_bench(tmp_path)
    out = tmp_path / "results"
    seen = []

    def stand_in(refs, hyps):
        seen.append((list(refs), list(hyps)))
        return {"cer_strict": 0.5, "cer_flex": 0.25, "cer_kana": 0.0}

    assert eval_bench.main(_eval_argv(tiny_nemo, bench, out), cer_fn=stand_in) == 0
    assert seen[0][0] == ["𠮷野家で今日は", "良い天気ですね"]

    raw = (out / "tiny" / "hyps.jsonl").read_bytes()
    assert b"\r\n" not in raw  # どのOSでもLF
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["u0", "u1"]
    assert rows[0]["reference"] == "𠮷野家で今日は"
    assert [r["hypothesis"] for r in rows] == seen[0][1]

    summary = json.loads((out / "tiny" / "summary.json").read_text(encoding="utf-8"))
    assert summary["cer"] == {"cer_strict": 50.0, "cer_flex": 25.0, "cer_kana": 0.0}
    assert summary["n_utterances"] == 2
    assert summary["published"] is None
    assert summary["scorer"] == "stand_in"
    assert summary["environment"]["backend"] == "torch"
    assert "tiny" in (out / "summary.md").read_text(encoding="utf-8")
    assert "tiny" in capsys.readouterr().out


def test_eval_bench_end_to_end_with_jacer(tiny_nemo, tmp_path):
    import json

    pytest.importorskip("jacer")
    import eval_bench

    bench = _write_tiny_bench(tmp_path)
    out = tmp_path / "results"
    assert eval_bench.main(_eval_argv(tiny_nemo, bench, out)) == 0
    summary = json.loads((out / "tiny" / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["cer"]) == {"cer_strict", "cer_flex", "cer_kana"}
    assert summary["scorer"] == "jacer"
    assert summary["jacer_metrics_version"]


def test_bench_speed_end_to_end(tiny_nemo, tmp_path):
    import json

    import bench_speed

    bench = _write_tiny_bench(tmp_path)
    out = tmp_path / "speed.json"
    argv = [
        "--model", str(tiny_nemo), "--backend", "torch", "--bench", str(bench),
        "--subset", "tiny", "--warmup", "1", "--out", str(out),
    ]  # fmt: skip
    assert bench_speed.main(argv) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["n_utterances"] == 1
    assert result["peak_rss_bytes"] >= result["peak_rss_bytes_after_load"] > 0
    assert result["peak_rss_source"] == benchlib.peak_memory()["peak_rss_source"]
    assert [u["id"] for u in result["per_utterance"]] == ["u0", "u1"]


def test_long_audio_reads_utf8_and_utf16_transcripts(tmp_path):
    import long_audio

    text = '{"text": "𠮷野家"}\n'
    for name, data in {
        "utf8.json": text.encode("utf-8"),
        "utf8-bom.json": text.encode("utf-8-sig"),
        "utf16.json": text.encode("utf-16"),  # BOM付き（PowerShell 5.1の>）
    }.items():
        (tmp_path / name).write_bytes(data)
        assert long_audio.read_text_any_bom(tmp_path / name) == text, name
