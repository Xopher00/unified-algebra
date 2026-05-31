"""pytest configuration for the explore layer."""

import os
import sys
import pytest

_HERE = os.path.dirname(__file__)

# Add support/python/ so `from backends import ...` works in arch.py files.
sys.path.insert(0, os.path.join(_HERE, "support", "python"))



def pytest_collect_file(parent, file_path):
    """Collect arch.py files as test modules."""
    if file_path.name == "arch.py":
        return pytest.Module.from_parent(parent, path=file_path)
