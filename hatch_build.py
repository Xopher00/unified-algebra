"""Hatchling build hook: downloads the Hydra kernel at install time.

The Hydra library (CategoricalData/hydra) ships its generated kernel
(core.py, reduction.py, etc.) in dist/python/hydra-kernel/ with no
pyproject.toml, so it can't be installed directly via pip/uv.

This hook resolves the commit that pip/uv will install for the runtime
(heads/python/) and downloads the matching kernel, so both halves are
guaranteed to come from the same commit.

TODO: Once CategoricalData/hydra creates a proper release tag, the runtime
dependency in pyproject.toml can be pinned to that tag, and this hook can
use the same tag to download the kernel. Currently using HEAD of main.
See: https://github.com/CategoricalData/hydra
"""

import io
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

HYDRA_REPO = "https://github.com/CategoricalData/hydra.git"
KERNEL_PATH = "dist/python/hydra-kernel/src/main/python/hydra/"


def _resolve_commit() -> str:
    """Resolve the HEAD commit of the Hydra repo — same one pip installs."""
    result = subprocess.run(
        ["git", "ls-remote", HYDRA_REPO, "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.split()[0]


class CustomBuildHook(BuildHookInterface):

    def initialize(self, version, build_data):
        kernel_dest = Path(self.root) / "src" / "hydra"

        if not kernel_dest.exists():
            commit = _resolve_commit()
            self._download_kernel(kernel_dest, commit)

        build_data["force_include"][str(kernel_dest)] = "hydra"

    def _download_kernel(self, dest: Path, commit: str):
        prefix = f"hydra-{commit}/{KERNEL_PATH}"
        url = f"https://github.com/CategoricalData/hydra/archive/{commit}.zip"
        print(f"Downloading Hydra kernel ({commit[:8]})...")

        with urllib.request.urlopen(url) as response:  # noqa: S310
            data = response.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if not name.startswith(prefix):
                    continue
                rel = name[len(prefix):]
                if not rel or rel.endswith("/"):
                    continue
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(zf.read(name))

        print("Hydra kernel installed.")
