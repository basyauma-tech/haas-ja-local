"""scripts/の評価・計測スクリプトの純粋な関数。"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchlib  # noqa: E402
import long_audio  # noqa: E402
import rebuild_bench  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_read_manifest_resolves_relative_audio_and_accepts_audio_filepath(tmp_path):
    write_jsonl(
        tmp_path / "s.jsonl",
        [
            {"id": "a", "audio": "audio/s/a.flac", "text": "あ", "duration": 1.5},
            {"audio_filepath": "/abs/b.flac", "text": "い"},
        ],
    )
    items = benchlib.read_manifest(tmp_path, "s")
    assert items[0] == benchlib.Utterance("a", tmp_path / "audio/s/a.flac", "あ", 1.5)
    assert items[1] == benchlib.Utterance("b", Path("/abs/b.flac"), "い", 0.0)


def test_read_manifest_limit(tmp_path):
    write_jsonl(tmp_path / "s.jsonl", [{"id": str(i), "audio": f"{i}.flac", "text": "x"} for i in range(5)])
    assert [u.id for u in benchlib.read_manifest(tmp_path, "s", limit=2)] == ["0", "1"]


def test_rtfx():
    assert benchlib.rtfx(10.0, 0.5) == 20.0
    assert benchlib.rtfx(10.0, 0.0) == float("inf")


def test_speed_stats_excludes_warmup():
    audio = [100.0, 10.0, 10.0, 20.0]
    proc = [100.0, 1.0, 0.5, 4.0]  # 先頭はウォームアップ（遅い）
    s = benchlib.speed_stats(audio, proc, warmup=1)
    assert s["n_utterances"] == 3
    assert s["audio_seconds"] == 40.0
    assert s["processing_seconds"] == 5.5
    assert s["rtfx_total"] == pytest.approx(40.0 / 5.5)
    assert s["rtfx_median"] == 10.0  # 10, 20, 5
    assert s["rtfx_mean"] == pytest.approx(35.0 / 3)
    assert s["seconds_per_utterance_median"] == 1.0


def test_speed_stats_rejects_all_warmup():
    with pytest.raises(ValueError):
        benchlib.speed_stats([1.0], [1.0], warmup=3)


def test_cer_percent():
    assert benchlib.cer_percent({"cer_strict": 0.0365, "cer_flex": 0.03, "cer_kana": 0.02, "x": 1}) == pytest.approx(
        {"cer_strict": 3.65, "cer_flex": 3.0, "cer_kana": 2.0}
    )


@pytest.mark.parametrize(
    ("value", "delta", "passed"),
    [(3.66, 0.0, True), (3.50, -0.15, True), (3.97, 0.30, True), (3.98, 0.31, False), (3.30, -0.35, False)],
)
def test_published_delta_measures_from_the_nearest_end(value, delta, passed):
    d = benchlib.published_delta("fleurs", value)
    assert d["delta_points"] == pytest.approx(delta)
    assert d["pass"] is passed
    assert d["published_cer_strict"] == [3.65, 3.67]


def test_published_delta_unknown_subset():
    assert benchlib.published_delta("jsut", 5.0) is None


def test_markdown_table_row():
    summary = {
        "subset": "fleurs",
        "n_utterances": 650,
        "audio_seconds": 7200.0,
        "cer": {"cer_strict": 3.7, "cer_flex": 3.1, "cer_kana": 2.0},
        "published": benchlib.published_delta("fleurs", 3.7),
        "rtfx": 81.25,
    }
    table = benchlib.markdown_table([summary])
    assert "| fleurs | 650 | 2.00 | 3.70 | 3.10 | 2.00 | 3.65〜3.67 | +0.03 | 81.2 |" in table


def test_rebuild_command_forwards_subsets_and_jvnv():
    cmd = rebuild_bench.build_command(
        Path("/d/rebuild.py"), subsets=["fleurs", "jvnv"], out=Path("o"), jvnv_zip=Path("j.zip")
    )
    assert cmd[1:] == ["-I", "/d/rebuild.py", "--out", "o", "--subset", "fleurs", "--subset", "jvnv", "--jvnv", "j.zip"]
    assert cmd[0] == sys.executable


def test_is_monotonic():
    assert long_audio.is_monotonic([0.0, 0.0, 1.0, 2.5])
    assert not long_audio.is_monotonic([0.0, 2.0, 1.0])
    assert long_audio.is_monotonic([])
