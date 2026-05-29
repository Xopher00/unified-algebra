"""pytest configuration for the explore harness."""

import os
import sys

# Add explore/ itself to path for strategies.py
sys.path.insert(0, os.path.dirname(__file__))
