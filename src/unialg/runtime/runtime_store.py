"""Per-run native tensor store for the binary-adapter execution path.

Native backend values (numpy arrays, etc.) live here during one CompiledProgram.run() call.
Hydra terms carry 16-byte UUID keys into and out of primitives; primitives look up and
deposit native values here via O(1) dict operations — no npy serialization between ops.

Lifetime: created once per BackendOps; reset() called at the start and end of each run().
"""

import uuid
from typing import Any


class RuntimeStore:
    """UUID-keyed store for native tensor values during one CompiledProgram.run() call."""

    def __init__(self):
        self._data: dict[bytes, Any] = {}

    def reset(self) -> None:
        self._data.clear()

    def put(self, native) -> bytes:
        """Store a native value and return its 16-byte UUID handle."""
        key = uuid.uuid4().bytes
        self._data[key] = native
        return key

    def get(self, key: bytes):
        """Retrieve a native value by its handle. Raises KeyError on miss."""
        return self._data[key]
