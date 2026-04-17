from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import viser

from nerfstudio.viewer.utils import CameraState, get_camera

from virtual_nerf_explorer.config import RenderConfig

RenderPhase = Literal["move", "static"]


@dataclass(slots=True)
class RenderRequest:
    camera_state: CameraState
    phase: RenderPhase


class ClientRenderWorker(threading.Thread):
    def __init__(
        self,
        *,
        client: viser.ClientHandle,
        pipeline,
        render_lock: threading.Lock,
        render_config: RenderConfig,
        image_format: str,
        on_render_status,
    ) -> None:
        super().__init__(daemon=True)
        self.client = client
        self.pipeline = pipeline
        self.render_lock = render_lock
        self.render_config = render_config
        self.image_format = image_format
        self.on_render_status = on_render_status
        self._trigger = threading.Event()
        self._request: RenderRequest | None = None
        self._running = True
        self._last_motion_time = 0.0

    def submit(self, request: RenderRequest) -> None:
        if request.phase == "move":
            self._last_motion_time = time.time()
        self._request = request
        self._trigger.set()

    def stop(self) -> None:
        self._running = False
        self._trigger.set()

    def _image_size(self, aspect: float, phase: RenderPhase) -> tuple[int, int]:
        if not np.isfinite(aspect) or aspect <= 0.0:
            aspect = 1.0
        max_res = self.render_config.static_max_res if phase == "static" else self.render_config.moving_max_res
        image_height = max_res
        image_width = int(round(image_height * aspect))
        if image_width > max_res:
            image_width = max_res
            image_height = int(round(image_width / aspect))
        return max(image_height, 32), max(image_width, 32)

    def _render_rgb(self, camera_state: CameraState, phase: RenderPhase) -> np.ndarray:
        image_height, image_width = self._image_size(camera_state.aspect, phase)
        camera = get_camera(camera_state, image_height, image_width).to(self.pipeline.device)
        model = self.pipeline.model
        with self.render_lock, torch.no_grad(), contextlib.nullcontext():
            was_training = model.training
            model.eval()
            try:
                outputs = model.get_outputs_for_camera(camera)
            finally:
                if was_training:
                    model.train()
        if "rgb" not in outputs:
            raise KeyError("Loaded model did not return an 'rgb' output for camera rendering.")
        rgb = outputs["rgb"].detach().clamp(0.0, 1.0)
        image = (rgb * 255.0).to(torch.uint8).cpu().numpy()
        if image.shape[-1] != 3:
            raise ValueError(f"Expected RGB output with 3 channels, got shape {tuple(image.shape)}")
        return image

    def _send_image(self, image: np.ndarray, phase: RenderPhase) -> None:
        quality = self.render_config.jpeg_quality_static if phase == "static" else self.render_config.jpeg_quality_moving
        self.client.scene.set_background_image(
            image,
            format=self.image_format,
            jpeg_quality=quality,
        )
        self.on_render_status(f"Rendered {image.shape[1]}x{image.shape[0]} ({phase})")

    def run(self) -> None:
        while self._running:
            if not self._trigger.wait(self.render_config.idle_sleep_seconds):
                if self._request is not None and (time.time() - self._last_motion_time) >= self.render_config.static_transition_seconds:
                    self.submit(RenderRequest(camera_state=self._request.camera_state, phase="static"))
                continue
            self._trigger.clear()
            request = self._request
            if request is None or not self._running:
                continue
            image = self._render_rgb(request.camera_state, request.phase)
            if not self._running:
                break
            self._send_image(image, request.phase)
