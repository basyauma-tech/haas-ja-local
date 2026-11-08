""".nemoからONNX用のディレクトリ（fp32/、int8/、config.json、tokenizer.model、preprocessor.npz）を作る。

torch_model.HaasTorchをfloat32で読み、torch.onnx.exportの従来方式（TorchScriptのトレース、opset 17）で
エンコーダとデコーダ＋結合の2本に書き出す。入出力名はNeMoのRNNT書き出しと同じにする。
ONNXグラフには前処理（log-mel）を含めないので、窓とメルフィルタバンクはpreprocessor.npzで配る。

出力の配置:

    out_dir/
      config.json、tokenizer.model、preprocessor.npz
      fp32/encoder.onnx、fp32/encoder.onnx.data、fp32/decoder_joint.onnx、fp32/SHA256SUMS
      int8/encoder.onnx（.dataが付くことがある）、int8/decoder_joint.onnx、int8/SHA256SUMS（quantize_int8）

検査はすべて書き込みの前に行う。書き出しはout_dir内の一時ディレクトリで行い、最後に置き換えるので、
途中で失敗したときは既存のファイルを変えない。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from haas_ja_local.architecture import subsampling_stages
from haas_ja_local.config import HaasConfig
from haas_ja_local.nemo_checkpoint import read_nemo
from haas_ja_local.torch_model import HaasTorch, build_torch_model

if TYPE_CHECKING:
    import onnx

OPSET = 17
# トレースに使うmelのフレーム数。時間軸は動的軸なので、推論時の長さはこれと違ってよい
TRACE_FRAMES = 32

ENCODER_INPUTS = ("audio_signal", "length")
ENCODER_OUTPUTS = ("outputs", "encoded_lengths")
DECODER_INPUTS = ("encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2")
DECODER_OUTPUTS = ("outputs", "prednet_lengths", "output_states_1", "output_states_2")

ENCODER_FILE = "encoder.onnx"
ENCODER_DATA = "encoder.onnx.data"
DECODER_FILE = "decoder_joint.onnx"
SUMS_FILE = "SHA256SUMS"
ROOT_FILES = ("config.json", "tokenizer.model", "preprocessor.npz")
FP32_FILES = (ENCODER_FILE, ENCODER_DATA, DECODER_FILE)

NOTE = "Unofficial conversion. Not affiliated with Third Intelligence."


class EncoderWrapper(nn.Module):
    """audio_signal [B,n_mels,T] f32、length [B] i64 → (outputs [B,d_model,T'], encoded_lengths [B] i64)。

    計算はtorch_model.ConformerEncoder.forwardと同じ。最長の有効長での切り詰めだけを、トレースで定数に
    ならないようテンソルの演算で書く（元の`int(lengths.max())`はトレース時の値に固定される）。
    単一発話（B=1）を前提とし、長さの異なる発話をまとめて渡したときの短い発話の出力は保証しない。
    """

    def __init__(self, model: HaasTorch) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.stages = subsampling_stages(model.config)

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder
        lengths = length.to(torch.int64)
        x = enc.pre_encode(audio_signal.transpose(1, 2), lengths)
        for _ in range(self.stages):
            lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        x = x.index_select(1, torch.arange(lengths.max(), device=x.device))
        x, pos_emb = enc.pos_enc(x)
        for layer in enc.layers:
            x = layer(x, pos_emb)
        return x.transpose(1, 2), lengths


class DecoderJointWrapper(nn.Module):
    """予測ネットワーク1ステップと結合。

    encoder_outputs [B,d_model,1] f32、targets [B,1] i32、target_length [B] i32、
    input_states_1・input_states_2 [L,B,kda_state_width] f32
    → (outputs [B,1,1,vocab_size+1+len(durations)], prednet_lengths [B] i32、output_states_1、output_states_2)。
    prednet_lengthsはtarget_lengthをそのまま返す（NeMoの書き出しと同じ出力の並びにするため）。
    """

    def __init__(self, model: HaasTorch) -> None:
        super().__init__()
        self.prediction = model.decoder.prediction
        self.joint = model.joint

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        targets: torch.Tensor,
        target_length: torch.Tensor,
        input_states_1: torch.Tensor,
        input_states_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g, rec, conv = self.prediction(targets[:, 0].to(torch.int64), input_states_1, input_states_2)
        logits = self.joint(encoder_outputs[:, :, 0], g)
        return logits.unsqueeze(1).unsqueeze(1), target_length, rec, conv


def onnx_metadata(config: HaasConfig, source_sha256: str) -> dict[str, str]:
    """両方のONNXのmetadata_propsに入れる値。sherpa-onnxのNeMo transducerが読むキーを含む。"""
    from haas_ja_local import __version__

    return {
        "vocab_size": str(config.vocab_size),
        "blank_id": str(config.blank_id),
        "model_type": "tdt",
        "durations": ",".join(str(d) for d in config.durations),
        "pred_rnn_layers": str(config.pred_layers),
        "pred_hidden": str(config.pred_hidden),
        "kda_state_width": str(config.kda_state_width),
        "normalize_type": "per_feature",
        "subsampling_factor": str(config.subsampling_factor),
        "feat_dim": str(config.n_mels),
        "frame_seconds": f"{config.frame_seconds:g}",
        "sample_rate": str(config.sample_rate),
        "source_repo": config.source_repo,
        "source_sha256": source_sha256,
        "converter": f"haas-ja-local {__version__}",
        "note": NOTE,
    }


def _export(module: nn.Module, args: tuple[torch.Tensor, ...], path: Path, **kwargs) -> None:
    with torch.no_grad(), warnings.catch_warnings():
        # 従来方式（dynamo=False）では次の3つが必ず出るので、これらだけを抑える。
        # - 従来方式を使うことへの非推奨の通知
        # - torch内部（setup_onnx_logging）が使う関数の非推奨の通知
        # - 刻み幅2のスライス（位置埋め込みのsin・cosの交互配置など）を定数畳み込みしないという通知。
        #   そのスライスは畳み込まれずに実行時に計算されるだけで、出力の値は変わらない
        warnings.filterwarnings(
            "ignore", message=r"You are using the legacy TorchScript-based ONNX export", category=DeprecationWarning
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The feature will be removed\. Please remove usage of this function",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore", message=r"Constant folding - Only steps=1 can be constant folded", category=UserWarning
        )
        torch.onnx.export(
            module,
            args,
            str(path),
            opset_version=OPSET,
            dynamo=False,
            do_constant_folding=True,
            **kwargs,
        )


def _export_encoder(model: HaasTorch, path: Path) -> None:
    c = model.config
    mel = torch.randn(1, c.n_mels, TRACE_FRAMES, generator=torch.Generator().manual_seed(0))
    _export(
        EncoderWrapper(model).eval(),
        (mel, torch.tensor([TRACE_FRAMES], dtype=torch.int64)),
        path,
        input_names=list(ENCODER_INPUTS),
        output_names=list(ENCODER_OUTPUTS),
        dynamic_axes={
            "audio_signal": {0: "batch", 2: "time"},
            "length": {0: "batch"},
            "outputs": {0: "batch", 2: "time_enc"},
            "encoded_lengths": {0: "batch"},
        },
    )


def _export_decoder_joint(model: HaasTorch, path: Path) -> None:
    c = model.config
    state = torch.zeros(c.pred_layers, 1, c.kda_state_width)
    _export(
        DecoderJointWrapper(model).eval(),
        (
            torch.zeros(1, c.d_model, 1),
            torch.full((1, 1), c.blank_id, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            state,
            state.clone(),
        ),
        path,
        input_names=list(DECODER_INPUTS),
        output_names=list(DECODER_OUTPUTS),
        dynamic_axes={
            "encoder_outputs": {0: "batch"},
            "targets": {0: "batch"},
            "target_length": {0: "batch"},
            "input_states_1": {1: "batch"},
            "input_states_2": {1: "batch"},
            "outputs": {0: "batch"},
            "prednet_lengths": {0: "batch"},
            "output_states_1": {1: "batch"},
            "output_states_2": {1: "batch"},
        },
    )


def _set_metadata(model: onnx.ModelProto, metadata: dict[str, str]) -> None:
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=key, value=value)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def write_sha256sums(directory: Path, names: Iterable[str]) -> None:
    """directory/SHA256SUMSを`sha256sum`と同じ書式（ハッシュ、空白2つ、ファイル名）で書く。"""
    lines = [f"{file_sha256(directory / name)}  {name}\n" for name in names]
    (directory / SUMS_FILE).write_text("".join(lines), encoding="utf-8", newline="\n")


def _install(stage: Path, dest: Path, names: Iterable[str]) -> None:
    """stage内のファイルをdestへ移し、権限を0644にそろえる（一時ディレクトリ内は0600になることがある）。"""
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        os.chmod(stage / name, 0o644)
        os.replace(stage / name, dest / name)


def export_onnx(nemo: Path, out_dir: Path) -> None:
    """.nemoを読み、out_dir/fp32/とconfig.json・tokenizer.model・preprocessor.npzを書く。

    チェックポイントのキーや形状が構成と合わなければ、何も書かずにValueError。
    """
    import onnx

    ckpt = read_nemo(Path(nemo))
    try:
        model = build_torch_model(ckpt, dtype=torch.float32)
    except RuntimeError as e:
        raise ValueError(f"checkpoint does not match the Haas-ja-v1.0 layout: {e}") from e
    metadata = onnx_metadata(ckpt.config, ckpt.sha256)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir, prefix=".export-") as tmp:
        stage = Path(tmp)
        raw = stage / "raw"
        fp32 = stage / "fp32"
        raw.mkdir()
        fp32.mkdir()

        # 従来方式は2GBを超えると重みをテンソルごとの別ファイルに書くので、読み直して1ファイルにまとめる
        _export_encoder(model, raw / ENCODER_FILE)
        encoder = onnx.load(raw / ENCODER_FILE, load_external_data=True)
        _set_metadata(encoder, metadata)
        onnx.save(
            encoder,
            fp32 / ENCODER_FILE,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ENCODER_DATA,
            size_threshold=1024,
        )
        del encoder

        _export_decoder_joint(model, raw / DECODER_FILE)
        decoder = onnx.load(raw / DECODER_FILE)
        _set_metadata(decoder, metadata)
        onnx.save(decoder, fp32 / DECODER_FILE)

        for name in (ENCODER_FILE, DECODER_FILE):
            onnx.checker.check_model(str(fp32 / name))
        write_sha256sums(fp32, FP32_FILES)

        ckpt.config.to_json(stage / "config.json")
        (stage / "tokenizer.model").write_bytes(ckpt.tokenizer_model)
        np.savez(
            stage / "preprocessor.npz",
            window=model.window_np.astype(np.float32),
            fb=model.fb_np.astype(np.float32),
        )

        _install(fp32, out_dir / "fp32", (*FP32_FILES, SUMS_FILE))
        _install(stage, out_dir, ROOT_FILES)


def quantize_int8(fp32_dir: Path, out_dir: Path, *, exclude_decoder: bool = False) -> None:
    """fp32_dirのONNXをONNX Runtimeの動的量子化（重みint8、per-channel）にかけ、out_dirへ書く。

    out_dirはint8/そのもの（例: bundle/int8）。exclude_decoder=Trueならデコーダはfp32を写す。
    reduce_rangeはFalse（x86での精度を見てTask 12で決める）。量子化前の前処理（quant_pre_process）はかけない。
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    fp32_dir = Path(fp32_dir)
    out_dir = Path(out_dir)
    for name in (ENCODER_FILE, DECODER_FILE):
        if not (fp32_dir / name).is_file():
            raise FileNotFoundError(f"fp32 model not found: {fp32_dir / name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir, prefix=".quantize-") as tmp:
        stage = Path(tmp)
        common = {"per_channel": True, "reduce_range": False, "weight_type": QuantType.QInt8}
        quantize_dynamic(fp32_dir / ENCODER_FILE, stage / ENCODER_FILE, use_external_data_format=True, **common)
        if exclude_decoder:
            shutil.copyfile(fp32_dir / DECODER_FILE, stage / DECODER_FILE)
        else:
            quantize_dynamic(fp32_dir / DECODER_FILE, stage / DECODER_FILE, use_external_data_format=False, **common)
        names = sorted(p.name for p in stage.iterdir() if p.is_file())
        write_sha256sums(stage, names)
        _install(stage, out_dir, (*names, SUMS_FILE))
