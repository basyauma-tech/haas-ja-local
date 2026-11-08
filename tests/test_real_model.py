"""実チェックポイントを使う煙テスト。HAAS_JA_NEMO / HAAS_JA_MODEL_DIRが無ければskipする。"""

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
