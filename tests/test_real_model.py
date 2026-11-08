"""実チェックポイントを使う煙テスト。HAAS_JA_NEMO / HAAS_JA_MODEL_DIRが無ければskipする。"""

import json
import os
from pathlib import Path

import pytest

from haas_ja_local.audio import load_audio
from haas_ja_local.decode import greedy_tdt
from haas_ja_local.features import log_mel

DATA = Path(__file__).parent / "data"
CLIP = DATA / "ja_short.flac"  # macOSのsay（Kyoko）で合成した約2秒の音声。読み上げ文はja_short.txt


@pytest.mark.real_model
def test_real_torch_transcribes_fixture_clip(real_nemo):
    from haas_ja_local.torch_model import load_torch_model

    model, tok = load_torch_model(real_nemo)
    mel, n = log_mel(load_audio(CLIP), model.window_np, model.fb_np, model.config)
    tokens = greedy_tdt(model, model.encode(mel, n))
    text = tok.decode([t.id for t in tokens])
    print(f"\nreference: {(DATA / 'ja_short.txt').read_text(encoding='utf-8').strip()}\nhypothesis: {text}")
    assert text  # 非空。内容の一致はTask 7の数値検証で見る


@pytest.mark.real_model
@pytest.mark.parametrize("name", sorted(json.loads((DATA / "expected_tokens.json").read_text(encoding="utf-8"))))
def test_real_torch_reproduces_expected_tokens(real_nemo, name):
    # FLEURS ja_jp testの2発話（CC-BY-4.0、出典はdocs/verification/2026-11-08-mac-numerics.md）。
    # 期待値は参照fp32 ONNXとトークン列が一致した自前torchの出力。音声はリポジトリに含めない
    clips_dir = os.environ.get("HAAS_JA_CLIPS_DIR")
    if not clips_dir:
        pytest.skip("HAAS_JA_CLIPS_DIR is not set")
    from haas_ja_local.torch_model import load_torch_model

    expected = json.loads((DATA / "expected_tokens.json").read_text(encoding="utf-8"))[name]
    model, _ = load_torch_model(real_nemo)
    mel, n = log_mel(load_audio(Path(clips_dir) / name), model.window_np, model.fb_np, model.config)
    assert [t.id for t in greedy_tdt(model, model.encode(mel, n))] == expected
