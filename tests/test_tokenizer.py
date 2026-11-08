from haas_ja_local.tokenizer import Tokenizer


def test_tokenizer_roundtrip(tiny_tokenizer_model):
    tok = Tokenizer(tiny_tokenizer_model)
    assert tok.vocab_size == 16
    ids = tok.encode("あい")  # テスト補助としてencodeも持つ（推論では使わない）
    assert tok.decode(ids) == "あい"
    assert "".join(tok.pieces(ids)).replace("▁", "") == "あい"


def test_decode_empty_is_empty_string(tiny_tokenizer_model):
    assert Tokenizer(tiny_tokenizer_model).decode([]) == ""


def test_tokenizer_loads_from_path(tiny_tokenizer_model, tmp_path):
    path = tmp_path / "tok.model"
    path.write_bytes(tiny_tokenizer_model)
    assert Tokenizer(path).vocab_size == 16
