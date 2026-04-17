from __future__ import annotations

from datetime import datetime
import re

import imageio.v3 as iio
import numpy as np
import viser


def _safe_filename(base_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "scene_capture"


def capture_scene_image(
    *,
    client: viser.ClientHandle,
    image: np.ndarray,
    base_name: str | None = None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = _safe_filename(base_name or "scene_capture")
    filename = f"{stem}_{timestamp}.png"
    payload = iio.imwrite("<bytes>", np.asarray(image), extension=".png")
    client.send_file_download(filename, payload)
    return filename
