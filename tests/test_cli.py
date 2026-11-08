import json

import numpy as np
import pytest
import soundfile as sf

from haas_ja_local.cli import main

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
