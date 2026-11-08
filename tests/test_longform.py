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


def test_quiet_gap_outside_search_range_is_ignored():
    # 無音が探索範囲（28〜32秒）の外の20秒にしかなければ、範囲内のどこかで切る
    audio = np.random.default_rng(0).standard_normal(SR * 70).astype(np.float32)
    audio[int(20.0 * SR) : int(20.4 * SR)] = 0.0
    first_cut = split_at_quiet_points(audio)[0][1]
    assert int(28.0 * SR) <= first_cut <= int(32.0 * SR)


def test_chunks_cover_audio_without_gaps_or_overlap():
    # 100秒乱数 → 断片が連続して全体を覆い、各断片は32秒以下、最後以外は15秒以上
    audio = np.random.default_rng(0).standard_normal(SR * 100).astype(np.float32)
    chunks = split_at_quiet_points(audio)
    assert len(chunks) >= 4
    assert chunks[0][0] == 0
    assert chunks[-1][1] == len(audio)
    for (_, end), (start, _) in itertools.pairwise(chunks):
        assert end == start
    for start, end in chunks:
        assert end - start <= 32 * SR
    for start, end in chunks[:-1]:
        assert end - start >= 15 * SR


def test_invalid_search_range_cuts_at_target():
    # chunk_seconds=1.0では探索下限start+15秒が上限target+2秒を超えるので、target（1秒ごと）で切る
    audio = np.random.default_rng(0).standard_normal(int(SR * 3.5)).astype(np.float32)
    chunks = split_at_quiet_points(audio, chunk_seconds=1.0)
    assert chunks == [(0, SR), (SR, 2 * SR), (2 * SR, 3 * SR), (3 * SR, len(audio))]
