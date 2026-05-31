"""pytest configuration for the explore layer."""

import os
import sys
import glob
import pytest

_HERE = os.path.dirname(__file__)

# Add explore/ itself so `from backends import ...` works in arch.py files.
sys.path.insert(0, _HERE)

# Add all generated backend dirs so generated modules are importable.
for _gen_dir in glob.glob(os.path.join(_HERE, "archs", "*", "generated", "*")):
    if os.path.isdir(_gen_dir):
        sys.path.insert(0, _gen_dir)


def pytest_collect_file(parent, file_path):
    """Collect arch.py files as test modules."""
    if file_path.name == "arch.py":
        return pytest.Module.from_parent(parent, path=file_path)
