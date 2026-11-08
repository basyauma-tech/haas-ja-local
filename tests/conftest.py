import io
import os
import tarfile
from pathlib import Path

import pytest

from haas_ja_local.config import HaasConfig

DATA = Path(__file__).parent / "data"
TOKENIZER_MEMBER = "./0123456789abcdef0123456789abcdef_tokenizer.model"


def make_state_dict(config: HaasConfig, seed: int = 0) -> dict:
    """expected_keysの全キーについて、形状どおりの乱数bf16テンソルを作る。"""
    import torch

    from haas_ja_local.nemo_checkpoint import expected_shapes

    gen = torch.Generator().manual_seed(seed)
    state_dict = {}
    for name, shape in sorted(expected_shapes(config).items()):
        if name.endswith("num_batches_tracked"):
            state_dict[name] = torch.tensor(0, dtype=torch.int64)
        elif name.endswith("running_var"):
            state_dict[name] = (torch.rand(shape, generator=gen) + 0.5).to(torch.bfloat16)
        elif name.startswith("preprocessor.featurizer."):
            # 窓とメルフィルタバンクは非負（負だとlog-melがNaNになる）
            state_dict[name] = (torch.rand(shape, generator=gen) * 0.1).to(torch.bfloat16)
        else:
            state_dict[name] = (torch.randn(shape, generator=gen) * 0.1).to(torch.bfloat16)
    return state_dict


def model_config_yaml(config: HaasConfig) -> str:
    """上流のmodel_config.yamlと同じ配置で、from_model_config_yamlが読むキーだけを書く。"""
    c = config
    durations = "".join(f"  - {d}\n" for d in c.durations)
    return (
        "preprocessor:\n"
        f"  features: {c.n_mels}\n"
        f"  n_fft: {c.n_fft}\n"
        "encoder:\n"
        f"  n_layers: {c.n_layers}\n"
        f"  d_model: {c.d_model}\n"
        f"  subsampling_factor: {c.subsampling_factor}\n"
        f"  subsampling_conv_channels: {c.subsampling_channels}\n"
        f"  ff_expansion_factor: {c.ff_expansion}\n"
        f"  n_heads: {c.n_heads}\n"
        f"  conv_kernel_size: {c.conv_kernel_size}\n"
        "decoder:\n"
        "  prednet:\n"
        f"    pred_hidden: {c.pred_hidden}\n"
        f"    pred_rnn_layers: {c.pred_layers}\n"
        f"    kda_num_heads: {c.kda_heads}\n"
        f"    kda_head_dim: {c.kda_head_dim}\n"
        f"    kda_conv_size: {c.kda_conv_size}\n"
        f"  vocab_size: {c.vocab_size}\n"
        "joint:\n"
        "  jointnet:\n"
        f"    joint_hidden: {c.joint_hidden}\n"
        "decoding:\n"
        "  durations:\n"
        f"{durations}"
        "  greedy:\n"
        f"    max_symbols: {c.max_symbols}\n"
    )


def write_nemo(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as tar:
        tar.addfile(_dir_info("./"))
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _dir_info(name: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    return info


@pytest.fixture(scope="session")
def tiny_config() -> HaasConfig:
    return HaasConfig(
        d_model=16,
        n_layers=2,
        n_heads=2,
        ff_expansion=2,
        conv_kernel_size=3,
        subsampling_channels=4,
        n_mels=8,
        pred_hidden=8,
        pred_layers=2,
        kda_heads=2,
        kda_head_dim=3,
        kda_conv_size=4,
        joint_hidden=12,
        vocab_size=16,
    )


@pytest.fixture(scope="session")
def tiny_tokenizer_model() -> bytes:
    import sentencepiece as spm

    lines = [line for line in (DATA / "tiny_corpus.txt").read_text(encoding="utf-8").splitlines() if line]
    buf = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(lines),
        model_writer=buf,
        vocab_size=16,
        model_type="bpe",
        byte_fallback=False,
        character_coverage=1.0,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        minloglevel=2,
    )
    return buf.getvalue()


@pytest.fixture
def tiny_state_dict(tiny_config) -> dict:
    return make_state_dict(tiny_config, seed=0)


@pytest.fixture(scope="session")
def tiny_nemo(tmp_path_factory, tiny_config, tiny_tokenizer_model) -> Path:
    import torch

    weights = io.BytesIO()
    torch.save(make_state_dict(tiny_config, seed=0), weights)
    members = {
        TOKENIZER_MEMBER: tiny_tokenizer_model,
        "./model_config.yaml": model_config_yaml(tiny_config).encode("utf-8"),
        "./model_weights.ckpt": weights.getvalue(),
    }
    return write_nemo(tmp_path_factory.mktemp("tiny") / "tiny.nemo", members)


@pytest.fixture
def real_nemo() -> Path:
    value = os.environ.get("HAAS_JA_NEMO")
    if not value:
        pytest.skip("HAAS_JA_NEMO is not set")
    return Path(value)


@pytest.fixture
def real_model_dir() -> Path:
    value = os.environ.get("HAAS_JA_MODEL_DIR")
    if not value:
        pytest.skip("HAAS_JA_MODEL_DIR is not set")
    return Path(value)
