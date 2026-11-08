"""asr-ja-bench（Hugging Faceのデータセットthird-intelligence/asr-ja-bench）を手元に再構築する。

データセットのリポジトリ（rebuild.pyとメタデータ）をsnapshot_downloadで取り、その中のrebuild.pyを
別プロセス（`python -I`）で実行する。FLEURSはHugging Faceから自動で取る。JVNVは公式サイトから
jvnv_ver1.zipを取得して--jvnv-zipで渡す。rebuild.pyは書き起こしのハッシュを照合し、不一致なら非0で終わる。

必要なもの: huggingface_hub・pyarrow・soundfile・numpy（`--extra eval`）と、ffmpegコマンド。

実行例:
    uv run --extra eval python scripts/rebuild_bench.py --subset fleurs jvnv \
        --jvnv-zip var/bench/jvnv_ver1.zip --out var/bench/asr-ja-bench-v1.1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

DATASET = "third-intelligence/asr-ja-bench"


def build_command(rebuild_py: Path, *, subsets: list[str], out: Path, jvnv_zip: Path | None) -> list[str]:
    command = [sys.executable, "-I", str(rebuild_py), "--out", str(out)]
    for subset in subsets:
        command += ["--subset", subset]
    if jvnv_zip is not None:
        command += ["--jvnv", str(jvnv_zip)]
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", nargs="+", default=["fleurs"], help="subsets to rebuild (default fleurs)")
    parser.add_argument("--jvnv-zip", type=Path, help="jvnv_ver1.zip from the JVNV site (needed for jvnv)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--revision", help="dataset revision (default: main)")
    args = parser.parse_args(argv)

    if "jvnv" in args.subset and (args.jvnv_zip is None or not args.jvnv_zip.exists()):
        parser.error("--subset jvnv needs --jvnv-zip pointing to jvnv_ver1.zip")

    from huggingface_hub import snapshot_download

    source = Path(snapshot_download(DATASET, repo_type="dataset", revision=args.revision))
    command = build_command(source / "rebuild.py", subsets=args.subset, out=args.out, jvnv_zip=args.jvnv_zip)
    print(" ".join(command), file=sys.stderr)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
