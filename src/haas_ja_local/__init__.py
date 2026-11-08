"""Haas-ja-v1.0をNeMoなしでローカル実行するための推論パッケージ。

torch・mlx・onnxruntimeはload_modelで選んだバックエンドだけを遅延して読む。
"""

from haas_ja_local.audio import load_audio
from haas_ja_local.config import HaasConfig
from haas_ja_local.decode import TimedToken
from haas_ja_local.pipeline import Transcript, load_model, transcribe

__version__ = "0.1.0"

__all__ = ["HaasConfig", "TimedToken", "Transcript", "__version__", "load_audio", "load_model", "transcribe"]
