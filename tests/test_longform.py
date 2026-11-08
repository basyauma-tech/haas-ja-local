import itertools

import numpy as np

from haas_ja_local.longform import split_at_quiet_points

SR = 16000


def test_short_audio_is_single_chunk():
    assert split_at_quiet_points(np.zeros(SR * 20, np.float32)) == [(0, SR * 20)]


def test_exactly_chunk_length_is_single_chunk():
    assert split_at_quiet_points(np.ones(SR * 30, np.float32)) == [(0, SR * 30)]


def test_empty_audio_has_no_chunks():
    assert split_at_quiet_points(np.zeros(0, np.float32)) == []


def test_cut_falls_in_quiet_gap():
    # 70秒: 29.0〜29.4秒だけ無音、他は乱数。最初の切れ目は29.0〜29.4秒の中に入る
    audio = np.random.default_rng(0).standard_normal(SR * 70).astype(np.float32)
    audio[int(29.0 * SR) : int(29.4 * SR)] = 0.0
    chunks = split_at_quiet_points(audio)
    first_cut = chunks[0][1]
    assert int(29.0 * SR) <= first_cut <= int(29.4 * SR)


def test_cut_is_center_of_quietest_window():
    # 窓長ちょうど（0.1秒）の無音が29.5秒から始まる → その窓の中心29.55秒で切る
    audio = np.random.default_rng(0).standard_normal(SR * 70).astype(np.float32)
    audio[int(29.5 * SR) : int(29.6 * SR)] = 0.0
    assert split_at_quiet_points(audio)[0] == (0, int(29.55 * SR))


def test_search_starts_at_half_chunk():
    # chunk 10秒・探索8秒なら探索下限はmax(start+5秒, target-8秒)=5秒（15秒ではない）。
    # 4.9〜5.1秒の無音のうち、5.0秒から始まる窓が範囲内で最初の無音窓になり、その中心5.05秒で切る
    audio = np.random.default_rng(0).standard_normal(SR * 30).astype(np.float32)
    audio[int(4.9 * SR) : int(5.1 * SR)] = 0.0
    chunks = split_at_quiet_points(audio, chunk_seconds=10.0, search_seconds=8.0)
    assert chunks[0] == (0, int(5.05 * SR))


def test_quiet_gap_outside_search_range_is_ignored():
    # 無音が探索範囲（28〜32秒）の外の20秒にしかなければ、範囲内のどこかで切る
    audio = np.random.default_rng(0).standard_normal(SR * 70).astype(np.float32)
    audio[int(20.0 * SR) : int(20.4 * SR)] = 0.0
    first_cut = split_at_quiet_points(audio)[0][1]
    assert int(28.0 * SR) <= first_cut <= int(32.0 * SR)


def test_chunks_cover_audio_without_gaps_or_overlap():
    # 100秒乱数 → 断片が連続して全体を覆い、各断片は32.05秒（target+探索+窓の半分）以下、最後以外は15秒以上
    audio = np.random.default_rng(0).standard_normal(SR * 100).astype(np.float32)
    chunks = split_at_quiet_points(audio)
    assert len(chunks) >= 4
    assert chunks[0][0] == 0
    assert chunks[-1][1] == len(audio)
    for (_, end), (start, _) in itertools.pairwise(chunks):
        assert end == start
    for start, end in chunks:
        assert end - start <= int(32.05 * SR)
    for start, end in chunks[:-1]:
        assert end - start >= 15 * SR


def test_zero_search_cuts_at_center_of_target_window():
    # 探索0秒では窓の開始位置がtargetの1点だけになり、その窓の中心（target+0.05秒）で切る
    audio = np.random.default_rng(0).standard_normal(int(SR * 3.5)).astype(np.float32)
    chunks = split_at_quiet_points(audio, chunk_seconds=1.0, search_seconds=0.0)
    assert [end for _, end in chunks] == [SR + 800, 2 * SR + 1600, 3 * SR + 2400, len(audio)]


def test_invalid_search_range_cuts_at_target():
    # 1.05秒・chunk 1秒・探索0秒: 下限target=1秒が上限len-窓長=0.95秒を超えるので、targetで切る
    audio = np.random.default_rng(0).standard_normal(int(SR * 1.05)).astype(np.float32)
    chunks = split_at_quiet_points(audio, chunk_seconds=1.0, search_seconds=0.0)
    assert chunks == [(0, SR), (SR, len(audio))]
