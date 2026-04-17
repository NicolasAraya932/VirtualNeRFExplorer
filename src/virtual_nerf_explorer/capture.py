from __future__ import annotations

from datetime import datetime
from io import BytesIO
import re

import imageio.v3 as iio
import numpy as np
import torch
import viser

from virtual_nerf_explorer.render import RenderExport


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


def capture_tensor_export(
    *,
    client: viser.ClientHandle,
    export: RenderExport,
    base_name: str | None = None,
    export_format: str = "npz",
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = _safe_filename(base_name or "scene_capture")
    export_format = export_format.lower()

    if export_format == "pt":
        filename = f"{stem}_{timestamp}.pt"
        payload_stream = BytesIO()
        torch.save(
            {
                "image": torch.from_numpy(export.image.copy()),
                "tensors": export.tensors,
                "metadata": export.metadata,
            },
            payload_stream,
        )
        payload = payload_stream.getvalue()
    else:
        filename = f"{stem}_{timestamp}.npz"
        payload_stream = BytesIO()
        arrays = {"image": export.image}
        arrays.update({key: value.numpy() for key, value in export.tensors.items()})
        arrays.update({f"meta_{key}": np.asarray(value) for key, value in export.metadata.items()})
        np.savez_compressed(payload_stream, **arrays)
        payload = payload_stream.getvalue()

    client.send_file_download(filename, payload)
    return filename
