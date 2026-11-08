import json

import numpy as np
import pytest
import soundfile as sf

from haas_ja_local.cli import main
from haas_ja_local.decode import TimedToken
from haas_ja_local.pipeline import Transcript

SR = 16000


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "clip.wav"
    audio = np.random.default_rng(0).standard_normal(SR * 2).astype(np.float32) * 0.1
    sf.write(path, audio, SR)
    return path


def test_transcribe_json_timestamps(tiny_nemo, wav, capsys):
    assert main(["transcribe", str(tiny_nemo), str(wav), "--json", "--timestamps"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["file"] == str(wav)
    assert isinstance(record["text"], str)
    assert record["tokens"]  # 乱数重みでもこの入力ではトークンが出る（下の検査を空振りさせない）
    for token in record["tokens"]:
        assert set(token) == {"text", "start", "end"}
        assert 0.0 <= token["start"] < token["end"] <= 2.0 + 0.08 * 4
        assert round(token["start"], 3) == token["start"]


def test_token_times_and_piece_text(tiny_nemo, wav, capsys, monkeypatch):
    # 2番目の断片（offset 29.95秒。浮動小数の誤差が出る値）のトークンで、
    # 時刻の式・duration 0の扱い・丸め・先頭の"▁"の除去を確かめる
    from haas_ja_local.tokenizer import Tokenizer

    monkeypatch.setattr(Tokenizer, "pieces", lambda self, ids: ["▁か" if i == 3 else "な" for i in ids])
    fixed = Transcript("かな", (TimedToken(3, 3, 0), TimedToken(4, 7, 3)), (29.95, 29.95))
    monkeypatch.setattr("haas_ja_local.cli.transcribe", lambda *a, **k: fixed)
    assert main(["transcribe", str(tiny_nemo), str(wav), "--json", "--timestamps"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["text"] == "かな"
    assert record["tokens"] == [
        {"text": "か", "start": 30.19, "end": 30.27},
        {"text": "な", "start": 30.51, "end": 30.75},
    ]


def test_transcribe_json_without_timestamps_has_no_tokens(tiny_nemo, wav, capsys):
    assert main(["transcribe", str(tiny_nemo), str(wav), "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert set(record) == {"file", "text"}


def test_transcribe_plain_text_prefixes_file_for_multiple_inputs(tiny_nemo, wav, capsys):
    assert main(["transcribe", str(tiny_nemo), str(wav)]) == 0
    single = capsys.readouterr().out.splitlines()
    assert len(single) == 1
    assert not single[0].startswith(str(wav))

    assert main(["transcribe", str(tiny_nemo), str(wav), str(wav), "--threads", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert all(line.startswith(f"{wav}\t") for line in lines)


def test_missing_model_exits_1_with_message(tmp_path, wav, capsys):
    assert main(["transcribe", str(tmp_path / "missing.nemo"), str(wav)]) == 1
    err = capsys.readouterr().err
    assert "missing.nemo" in err
    assert len(err.strip().splitlines()) == 1


def test_missing_audio_exits_1_with_message(tiny_nemo, tmp_path, capsys):
    assert main(["transcribe", str(tiny_nemo), str(tmp_path / "missing.wav")]) == 1
    err = capsys.readouterr().err
    assert "missing.wav" in err
    assert len(err.strip().splitlines()) == 1


def test_bad_arguments_exit_2(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["transcribe", "model.nemo", "a.wav", "--backend", "tensorflow"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("argv", "variant"), [([], None), (["--variant", "int8"], "int8"), (["--variant", "fp32"], "fp32")]
)
def test_variant_flag_is_forwarded_to_load_model(tiny_nemo, wav, monkeypatch, argv, variant):
    seen = {}

    def fake_load_model(path, **kwargs):
        seen.update(kwargs)
        raise FileNotFoundError("stop here")

    monkeypatch.setattr("haas_ja_local.cli.load_model", fake_load_model)
    assert main(["transcribe", str(tiny_nemo), str(wav), *argv]) == 1
    assert seen["variant"] == variant


def test_variant_with_wrong_backend_exits_1_with_message(tiny_nemo, wav, capsys):
    assert main(["transcribe", str(tiny_nemo), str(wav), "--variant", "int8"]) == 1
    err = capsys.readouterr().err
    assert "variant" in err
    assert len(err.strip().splitlines()) == 1


def test_bad_variant_exits_2():
    with pytest.raises(SystemExit) as exc:
        main(["transcribe", "model", "a.wav", "--variant", "fp16"])
    assert exc.value.code == 2


class _FakeStdout:
    def __init__(self):
        self.calls = []

    def reconfigure(self, **kwargs):
        self.calls.append(kwargs)


@pytest.mark.parametrize(
    ("platform", "expected"), [("win32", [{"encoding": "utf-8", "errors": "replace"}]), ("linux", [])]
)
def test_main_makes_stdout_utf8_only_on_windows(monkeypatch, platform, expected):
    # Windowsの標準出力は既定でcp932などになり、書き起こしの文字が化けるか落ちる
    import sys

    fake = _FakeStdout()
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "stdout", fake)
    with pytest.raises(SystemExit):
        main(["--help-does-not-exist"])
    assert fake.calls == expected
