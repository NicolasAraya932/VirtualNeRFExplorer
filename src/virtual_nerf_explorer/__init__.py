"""VirtualNeRFExplorer package."""

from __future__ import annotations

import os

# Nerfstudio imports can trigger Torch Inductor compile worker processes on import.
# For this viewer they are unnecessary and make Ctrl+C shutdown noisy.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

__all__ = ["__version__"]
__version__ = "0.2.0"
