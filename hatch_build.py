"""Hatchling build hook: copies the Hydra kernel from the local repo.

Replaces the GitHub-download approach with a direct copy from the local
CategoricalData/hydra clone at /home/scanbot/hydra. Kernel files are
generated there by: bin/sync.sh --hosts python --targets python
"""

import shutil
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

LOCAL_HYDRA_KERNEL = Path("/home/scanbot/hydra/dist/python/hydra-kernel/src/main/python/hydra")


class CustomBuildHook(BuildHookInterface):

    def initialize(self, version, build_data):
        kernel_dest = Path(self.root) / "src" / "hydra"

        if kernel_dest.exists():
            shutil.rmtree(kernel_dest)

        shutil.copytree(LOCAL_HYDRA_KERNEL, kernel_dest)
        print(f"Hydra kernel copied from {LOCAL_HYDRA_KERNEL}")

        build_data["force_include"][str(kernel_dest)] = "hydra"
