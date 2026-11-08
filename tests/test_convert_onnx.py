import hashlib
import shutil
import stat
import sys

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")

from haas_ja_local import __version__  # noqa: E402
from haas_ja_local.architecture import encoded_length  # noqa: E402
from haas_ja_local.cli import main  # noqa: E402
from haas_ja_local.config import HaasConfig  # noqa: E402
from haas_ja_local.convert.onnx import (  # noqa: E402
    DECODER_INPUTS,
    DECODER_OUTPUTS,
    ENCODER_INPUTS,
    ENCODER_OUTPUTS,
    TRACE_FRAMES,
    export_onnx,
    quantize_int8,
)
from haas_ja_local.nemo_checkpoint import read_nemo  # noqa: E402
from haas_ja_local.onnx_model import HaasONNX  # noqa: E402
from haas_ja_local.torch_model import load_torch_model  # noqa: E402
from tests.conftest import TOKENIZER_MEMBER, make_state_dict, model_config_yaml, write_nemo  # noqa: E402

# float32同士の比較。演算の順序がtorchとONNX Runtimeで違っても通る幅
RTOL = ATOL = 1e-4

FP32_FILES = {"encoder.onnx", "encoder.onnx.data", "decoder_joint.onnx", "SHA256SUMS"}
ROOT_FILES = {"config.json", "tokenizer.model", "preprocessor.npz"}


@pytest.fixture(scope="module")
def tiny_onnx_dir(tiny_nemo, tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny-onnx")
    export_onnx(tiny_nemo, out)
    return out


@pytest.fixture(scope="module")
def parity_onnx_dir(parity_nemo, tmp_path_factory):
    out = tmp_path_factory.mktemp("parity-onnx")
    export_onnx(parity_nemo, out)
    return out


@pytest.fixture(scope="module")
def parity_torch(parity_nemo):
    model, _ = load_torch_model(parity_nemo)
    return model


@pytest.fixture(scope="module")
def parity_sessions(parity_onnx_dir):
    def session(name):
        return ort.InferenceSession(str(parity_onnx_dir / "fp32" / name), providers=["CPUExecutionProvider"])

    return session("encoder.onnx"), session("decoder_joint.onnx")


def _mel(config, n_frames, seed=0):
    return np.random.default_rng(seed).standard_normal((config.n_mels, n_frames)).astype(np.float32)


def _run_encoder(sess, mel, length):
    return sess.run(None, {"audio_signal": mel[None], "length": np.array([length], np.int64)})


def _metadata(model) -> dict[str, str]:
    return {p.key: p.value for p in model.metadata_props}


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_writes_expected_files(tiny_onnx_dir):
    assert {p.name for p in tiny_onnx_dir.iterdir()} == ROOT_FILES | {"fp32"}
    assert {p.name for p in (tiny_onnx_dir / "fp32").iterdir()} == FP32_FILES
    if sys.platform != "win32":
        for path in [*tiny_onnx_dir.glob("*.*"), *(tiny_onnx_dir / "fp32").iterdir()]:
            assert stat.S_IMODE(path.stat().st_mode) == 0o644, path


def test_export_io_names_types_and_opset(tiny_onnx_dir):
    enc = onnx.load(tiny_onnx_dir / "fp32" / "encoder.onnx", load_external_data=False)
    dec = onnx.load(tiny_onnx_dir / "fp32" / "decoder_joint.onnx")
    assert [i.name for i in enc.graph.input] == ["audio_signal", "length"] == list(ENCODER_INPUTS)
    assert [o.name for o in enc.graph.output] == ["outputs", "encoded_lengths"] == list(ENCODER_OUTPUTS)
    assert (
        [i.name for i in dec.graph.input]
        == [
            "encoder_outputs",
            "targets",
            "target_length",
            "input_states_1",
            "input_states_2",
        ]
        == list(DECODER_INPUTS)
    )
    assert (
        [o.name for o in dec.graph.output]
        == [
            "outputs",
            "prednet_lengths",
            "output_states_1",
            "output_states_2",
        ]
        == list(DECODER_OUTPUTS)
    )

    def elem_types(values):
        return [v.type.tensor_type.elem_type for v in values]

    t = onnx.TensorProto
    assert elem_types(enc.graph.input) == [t.FLOAT, t.INT64]
    assert elem_types(enc.graph.output) == [t.FLOAT, t.INT64]
    assert elem_types(dec.graph.input) == [t.FLOAT, t.INT32, t.INT32, t.FLOAT, t.FLOAT]
    assert elem_types(dec.graph.output) == [t.FLOAT, t.INT32, t.FLOAT, t.FLOAT]
    for model in (enc, dec):
        assert {o.domain: o.version for o in model.opset_import}[""] == 17


def test_encoder_weights_are_in_one_external_file_and_decoder_is_single_file(tiny_onnx_dir):
    enc = onnx.load(tiny_onnx_dir / "fp32" / "encoder.onnx", load_external_data=False)
    locations = set()
    n_external = 0
    for init in enc.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            n_external += 1
            locations |= {e.value for e in init.external_data if e.key == "location"}
    assert n_external > 0
    assert locations == {"encoder.onnx.data"}
    dec = onnx.load(tiny_onnx_dir / "fp32" / "decoder_joint.onnx", load_external_data=False)
    assert all(i.data_location != onnx.TensorProto.EXTERNAL for i in dec.graph.initializer)
    onnx.checker.check_model(str(tiny_onnx_dir / "fp32" / "encoder.onnx"))
    onnx.checker.check_model(str(tiny_onnx_dir / "fp32" / "decoder_joint.onnx"))


def test_export_metadata_on_both_graphs(tiny_onnx_dir, tiny_nemo, tiny_config):
    expected = {
        "vocab_size": str(tiny_config.vocab_size),
        "blank_id": str(tiny_config.blank_id),
        "model_type": "tdt",
        "durations": "0,1,2,3,4",
        "pred_rnn_layers": str(tiny_config.pred_layers),
        "pred_hidden": str(tiny_config.pred_hidden),
        "kda_state_width": str(tiny_config.kda_state_width),
        "normalize_type": "per_feature",
        "subsampling_factor": "8",
        "feat_dim": str(tiny_config.n_mels),
        "frame_seconds": "0.08",
        "sample_rate": "16000",
        "source_repo": "third-intelligence/Haas-ja-v1.0",
        "source_sha256": _sha256(tiny_nemo),
        "converter": f"haas-ja-local {__version__}",
        "note": "Unofficial conversion. Not affiliated with Third Intelligence.",
    }
    for name in ("encoder.onnx", "decoder_joint.onnx"):
        model = onnx.load(tiny_onnx_dir / "fp32" / name, load_external_data=False)
        assert _metadata(model) == expected, name


def test_sha256sums_match_files(tiny_onnx_dir):
    fp32 = tiny_onnx_dir / "fp32"
    lines = (fp32 / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    entries = {}
    for line in lines:
        digest, name = line.split("  ")
        entries[name] = digest
    assert set(entries) == FP32_FILES - {"SHA256SUMS"}
    for name, digest in entries.items():
        assert digest == _sha256(fp32 / name), name


def test_bundle_config_tokenizer_and_preprocessor(tiny_onnx_dir, tiny_nemo, tiny_tokenizer_model):
    ckpt = read_nemo(tiny_nemo)
    assert HaasConfig.from_json(tiny_onnx_dir / "config.json") == ckpt.config
    assert (tiny_onnx_dir / "tokenizer.model").read_bytes() == tiny_tokenizer_model
    model, _ = load_torch_model(tiny_nemo)
    with np.load(tiny_onnx_dir / "preprocessor.npz") as npz:
        assert set(npz.files) == {"window", "fb"}
        assert npz["window"].dtype == npz["fb"].dtype == np.float32
        np.testing.assert_array_equal(npz["window"], model.window_np)
        np.testing.assert_array_equal(npz["fb"], model.fb_np)


def test_export_rejects_mismatched_checkpoint_without_writing(tiny_config, tiny_tokenizer_model, tmp_path):
    import io

    state_dict = make_state_dict(tiny_config, seed=0)
    state_dict["joint.enc.weight"] = state_dict["joint.enc.weight"][:, :-1]
    weights = io.BytesIO()
    torch.save(state_dict, weights)
    bad = write_nemo(
        tmp_path / "bad.nemo",
        {
            TOKENIZER_MEMBER: tiny_tokenizer_model,
            "./model_config.yaml": model_config_yaml(tiny_config).encode("utf-8"),
            "./model_weights.ckpt": weights.getvalue(),
        },
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=r"joint\.enc\.weight"):
        export_onnx(bad, out)
    assert not out.exists() or not any(out.iterdir())


@pytest.mark.parametrize(
    ("t_pad", "length"),
    [
        (TRACE_FRAMES, TRACE_FRAMES),  # 書き出し時と同じ長さ
        (48, 48),  # 書き出し時と違う長さ（動的軸）
        (48, 37),  # 0埋めあり。8の倍数でない有効長
        (96, 81),
        (16, 1),
    ],
)
def test_onnx_encoder_matches_torch(parity_sessions, parity_torch, parity_config, t_pad, length):
    enc_sess, _ = parity_sessions
    mel = _mel(parity_config, t_pad, seed=t_pad + length)
    outputs, encoded_lengths = _run_encoder(enc_sess, mel, length)
    n_enc = encoded_length(length, parity_config)
    assert encoded_lengths.dtype == np.int64
    assert encoded_lengths.tolist() == [n_enc]
    # 自己注意の前で最長の有効長に切り詰める（torch_model.ConformerEncoderと同じ）
    assert outputs.shape == (1, parity_config.d_model, n_enc)
    expected = parity_torch.encode(mel, length).numpy()  # [T',d_model]
    np.testing.assert_allclose(outputs[0].T, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("length", [33, 37, 41, 47])
def test_onnx_encoder_ignores_padding_frames(parity_sessions, parity_config, length):
    # サブサンプリングの時間マスクがONNXでもlengthから作られる
    enc_sess, _ = parity_sessions
    mel = _mel(parity_config, 64, seed=1)
    other = mel.copy()
    other[:, length:] = np.random.default_rng(2).standard_normal((parity_config.n_mels, 64 - length))
    np.testing.assert_array_equal(_run_encoder(enc_sess, mel, length)[0], _run_encoder(enc_sess, other, length)[0])


def test_onnx_decoder_steps_match_torch(parity_sessions, parity_torch, parity_config):
    # blank → トークン → トークンの3ステップ。torchとONNXの状態はそれぞれ自分の出力を次へ渡す
    _, dec_sess = parity_sessions
    mel = _mel(parity_config, 64, seed=3)
    enc = parity_torch.encode(mel, 64)  # [T',d_model]
    t_state = parity_torch.init_state()
    o_state = tuple(s.numpy() for s in parity_torch.init_state())
    for t, token in ((0, parity_config.blank_id), (1, 3), (2, 7)):
        t_logits, t_state = parity_torch.step(enc, t, token, t_state)
        out = dec_sess.run(
            None,
            {
                "encoder_outputs": enc[t].numpy()[None, :, None],
                "targets": np.array([[token]], np.int32),
                "target_length": np.array([1], np.int32),
                "input_states_1": o_state[0],
                "input_states_2": o_state[1],
            },
        )
        logits, prednet_lengths, s1, s2 = out
        n_out = parity_config.vocab_size + 1 + len(parity_config.durations)
        assert logits.shape == (1, 1, 1, n_out)
        assert prednet_lengths.dtype == np.int32 and prednet_lengths.tolist() == [1]
        np.testing.assert_allclose(logits[0, 0, 0], t_logits, rtol=RTOL, atol=ATOL)
        for got, expected in zip((s1, s2), t_state, strict=True):
            assert got.shape == (parity_config.pred_layers, 1, parity_config.kda_state_width)
            np.testing.assert_allclose(got, expected.numpy(), rtol=RTOL, atol=ATOL)
        o_state = (s1, s2)
    assert o_state[0].any() and o_state[1].any()  # 0のままの比較で空振りしない


def _copy_bundle(src, dst):
    shutil.copytree(src, dst)
    return dst


def test_quantize_int8_runs_on_tiny(tiny_onnx_dir, tiny_config, tmp_path):
    bundle = _copy_bundle(tiny_onnx_dir, tmp_path / "bundle")
    quantize_int8(bundle / "fp32", bundle / "int8")
    int8 = bundle / "int8"
    names = {p.name for p in int8.iterdir()}
    assert {"encoder.onnx", "decoder_joint.onnx", "SHA256SUMS"} <= names
    sums = dict(reversed(line.split("  ")) for line in (int8 / "SHA256SUMS").read_text(encoding="utf-8").splitlines())
    assert set(sums) == names - {"SHA256SUMS"}
    for name, digest in sums.items():
        assert digest == _sha256(int8 / name), name

    fp32_meta = _metadata(onnx.load(bundle / "fp32" / "encoder.onnx", load_external_data=False))
    for name in ("encoder.onnx", "decoder_joint.onnx"):
        model = onnx.load(int8 / name)
        ops = {n.op_type for n in model.graph.node}
        assert ops & {"MatMulInteger", "DynamicQuantizeLinear", "ConvInteger"}, name  # 量子化されている
        # 量子化器はonnx.infer=onnxruntime.quantを足す。元のキーと値はすべて残る
        assert _metadata(model).items() >= fp32_meta.items(), name
        assert {o.domain: o.version for o in model.opset_import}[""] == 17

    model = HaasONNX.load(bundle, variant="int8")
    assert model.variant == "int8"
    mel = _mel(tiny_config, 48)
    enc = model.encode(mel, 45)
    assert enc.shape == (encoded_length(45, tiny_config), tiny_config.d_model)
    assert enc.dtype == np.float32 and np.isfinite(enc).all()
    logits, _ = model.step(enc, 0, tiny_config.blank_id, model.init_state())
    assert np.isfinite(logits).all()


def test_quantize_int8_exclude_decoder_copies_fp32(tiny_onnx_dir, tmp_path):
    bundle = _copy_bundle(tiny_onnx_dir, tmp_path / "bundle")
    quantize_int8(bundle / "fp32", bundle / "int8", exclude_decoder=True)
    assert (bundle / "int8" / "decoder_joint.onnx").read_bytes() == (
        bundle / "fp32" / "decoder_joint.onnx"
    ).read_bytes()
    enc_ops = {n.op_type for n in onnx.load(bundle / "int8" / "encoder.onnx").graph.node}
    assert "MatMulInteger" in enc_ops


def test_cli_convert_onnx(tiny_nemo, tmp_path, capsys):
    out = tmp_path / "onnx"
    assert main(["convert-onnx", "--nemo", str(tiny_nemo), "--out", str(out), "--int8"]) == 0
    printed = capsys.readouterr().out
    assert str(out / "fp32" / "encoder.onnx") in printed
    assert str(out / "int8" / "encoder.onnx") in printed
    assert f"source_sha256: {_sha256(tiny_nemo)}" in printed
    assert (out / "int8" / "decoder_joint.onnx").is_file()
    assert {p.name for p in out.iterdir()} == ROOT_FILES | {"fp32", "int8"}


def test_cli_convert_onnx_without_int8_and_exclude_decoder(tiny_nemo, tmp_path, capsys):
    out = tmp_path / "onnx"
    assert main(["convert-onnx", "--nemo", str(tiny_nemo), "--out", str(out)]) == 0
    assert not (out / "int8").exists()
    out2 = tmp_path / "onnx2"
    assert main(["convert-onnx", "--nemo", str(tiny_nemo), "--out", str(out2), "--int8", "--exclude-decoder"]) == 0
    assert (out2 / "int8" / "decoder_joint.onnx").read_bytes() == (out2 / "fp32" / "decoder_joint.onnx").read_bytes()


def test_cli_convert_onnx_missing_nemo(tmp_path, capsys):
    assert main(["convert-onnx", "--nemo", str(tmp_path / "none.nemo"), "--out", str(tmp_path / "o")]) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_exclude_decoder_requires_int8(tiny_nemo, tmp_path):
    with pytest.raises(SystemExit):
        main(["convert-onnx", "--nemo", str(tiny_nemo), "--out", str(tmp_path / "o"), "--exclude-decoder"])
