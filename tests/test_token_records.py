"""token_records（CLIの--timestampsが出すトークンごとの文字列と時刻）。"""

import pytest

from haas_ja_local.decode import TimedToken
from haas_ja_local.pipeline import token_records
from haas_ja_local.tokenizer import Tokenizer

FRAME = 0.08


class FakeTokenizer:
    """idからpieceを引く代理。decodeはバイトpieceをUTF-8として復元し、"▁"を空白にする。"""

    def __init__(self, pieces: dict[int, str]) -> None:
        self._pieces = pieces

    def pieces(self, ids):
        return [self._pieces[i] for i in ids]

    def decode(self, ids):
        out = bytearray()
        for piece in self.pieces(ids):
            if piece.startswith("<0x") and piece.endswith(">"):
                out.append(int(piece[3:-1], 16))
            else:
                out.extend(piece.replace("▁", " ").encode("utf-8"))
        return out.decode("utf-8", errors="replace")


def test_byte_fallback_pieces_merge_into_one_record():
    tok = FakeTokenizer({1: "▁", 2: "<0xE9>", 3: "<0xAC>", 4: "<0xB1>", 5: "▁今", 6: "日"})
    tokens = (
        TimedToken(1, 0, 1),
        TimedToken(2, 2, 0),
        TimedToken(3, 2, 0),
        TimedToken(4, 3, 2),
        TimedToken(5, 6, 1),
        TimedToken(6, 7, 0),
    )
    records = token_records(tokens, (0.0,) * len(tokens), tok, FRAME)
    assert records == [
        {"text": "鬱", "start": 0.16, "end": 0.40},  # 先頭<0xE9>の開始〜末尾<0xB1>の終了（3+2フレーム）
        {"text": "今", "start": 0.48, "end": 0.56},  # 先頭の"▁"は消す
        {"text": "日", "start": 0.56, "end": 0.64},  # duration 0は1フレームとして扱う
    ]


def test_byte_group_does_not_cross_chunks():
    # 断片（offset）が変わればバイト列のまとまりも切る
    tok = FakeTokenizer({2: "<0xE9>", 3: "<0xAC>", 4: "<0xB1>"})
    tokens = (TimedToken(2, 0, 1), TimedToken(3, 0, 1), TimedToken(4, 0, 1))
    records = token_records(tokens, (0.0, 30.0, 30.0), tok, FRAME)
    assert len(records) == 2
    assert [(r["start"], r["end"]) for r in records] == [(0.0, 0.08), (30.0, 30.08)]


def test_internal_word_boundary_becomes_space_and_empty_records_drop():
    tok = FakeTokenizer({1: "▁", 2: "▁a▁b", 3: "c"})
    tokens = (TimedToken(1, 0, 1), TimedToken(2, 1, 1), TimedToken(3, 2, 1))
    records = token_records(tokens, (0.0, 0.0, 0.0), tok, FRAME)
    assert [r["text"] for r in records] == ["a b", "c"]


def test_times_are_rounded_to_milliseconds():
    tok = FakeTokenizer({3: "▁か", 4: "な"})
    tokens = (TimedToken(3, 3, 0), TimedToken(4, 7, 3))
    assert token_records(tokens, (29.95, 29.95), tok, FRAME) == [
        {"text": "か", "start": 30.19, "end": 30.27},
        {"text": "な", "start": 30.51, "end": 30.75},
    ]


def test_tiny_tokenizer_records_match_decoded_text(tiny_tokenizer_model):
    tok = Tokenizer(tiny_tokenizer_model)
    ids = [i for i in range(1, tok.vocab_size)]
    tokens = tuple(TimedToken(i, n, 1) for n, i in enumerate(ids))
    records = token_records(tokens, (0.0,) * len(tokens), tok, FRAME)
    assert records
    assert all(r["text"] and not r["text"].startswith(" ") and "▁" not in r["text"] for r in records)
    assert "".join(r["text"] for r in records) == tok.decode(ids).replace(" ", "")


def test_lengths_must_match():
    with pytest.raises(ValueError):
        token_records((TimedToken(1, 0, 1),), (), FakeTokenizer({1: "a"}), FRAME)
