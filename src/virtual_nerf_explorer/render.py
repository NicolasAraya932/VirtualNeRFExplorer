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
from nerfstudio.utils import colormaps

from virtual_nerf_explorer.config import RenderConfig

RenderPhase = Literal["move", "static"]
RenderMode = Literal["rgb", "depth", "accumulation"]


@dataclass(slots=True)
class RenderRequest:
    camera_state: CameraState
    phase: RenderPhase
    render_mode: RenderMode
    depth_quantile: float


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
        super().__init__(daemon=False)
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
        self._latest_image_lock = threading.Lock()
        self._latest_image: np.ndarray | None = None

    def submit(self, request: RenderRequest) -> None:
        if request.phase == "move":
            self._last_motion_time = time.time()
        self._request = request
        self._trigger.set()

    def stop(self) -> None:
        self._running = False
        self._trigger.set()

    def get_latest_image(self) -> np.ndarray | None:
        with self._latest_image_lock:
            if self._latest_image is None:
                return None
            return self._latest_image.copy()

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

    def _approximate_depth_quantile(
        self, outputs: dict[str, torch.Tensor], depth_quantile: float
    ) -> torch.Tensor | None:
        quantile = float(np.clip(depth_quantile, 1e-3, 1.0 - 1e-3))
        weights = outputs.get("weights", None)
        steps = outputs.get("t_mid", None)

        if weights is None:
            density = outputs.get("density", None)
            if density is None or steps is None:
                return None
            density = density.detach()
            steps = steps.detach()
            if density.shape[-1] == 1:
                density = density[..., 0]
            if steps.shape[-1] == 1:
                steps = steps[..., 0]
            if steps.shape[-1] == 1:
                deltas = torch.ones_like(steps)
            else:
                deltas = torch.empty_like(steps)
                deltas[..., :-1] = steps[..., 1:] - steps[..., :-1]
                deltas[..., -1] = deltas[..., -2]
                deltas = torch.clamp(deltas, min=1e-6)
            alphas = 1.0 - torch.exp(-torch.relu(density) * deltas)
            transmittance = torch.cumprod(
                torch.cat([torch.ones_like(alphas[..., :1]), 1.0 - alphas + 1e-10], dim=-1),
                dim=-1,
            )[..., :-1]
            weights = (alphas * transmittance)[..., None]
        else:
            weights = weights.detach()
            if steps is None:
                return None
            steps = steps.detach()

        if weights.shape[-1] == 1:
            weights_1d = weights[..., 0]
        else:
            weights_1d = weights
        if steps.shape[-1] == 1:
            steps_1d = steps[..., 0]
        else:
            steps_1d = steps

        cumulative_weights = torch.cumsum(weights_1d, dim=-1).contiguous()
        split = torch.full((*weights_1d.shape[:-1], 1), quantile, device=weights_1d.device, dtype=weights_1d.dtype)
        depth_index = torch.searchsorted(cumulative_weights, split, side="left")
        depth_index = torch.clamp(depth_index, 0, steps_1d.shape[-1] - 1)
        return torch.gather(steps_1d, dim=-1, index=depth_index)

    def _convert_output_to_image(
        self,
        outputs: dict[str, torch.Tensor],
        render_mode: RenderMode,
        depth_quantile: float,
    ) -> np.ndarray:
        if render_mode == "rgb":
            if "rgb" not in outputs:
                raise KeyError("Loaded model did not return an 'rgb' output for camera rendering.")
            image_tensor = outputs["rgb"].detach().clamp(0.0, 1.0)
        elif render_mode == "depth":
            depth_tensor = self._approximate_depth_quantile(outputs, depth_quantile)
            if depth_tensor is None:
                if "depth" not in outputs:
                    raise KeyError("Loaded model did not return a 'depth' output for camera rendering.")
                depth_tensor = outputs["depth"].detach()
            image_tensor = colormaps.apply_depth_colormap(
                depth_tensor,
                accumulation=outputs.get("accumulation", None),
            ).clamp(0.0, 1.0)
        else:
            if "accumulation" not in outputs:
                raise KeyError("Loaded model did not return an 'accumulation' output for camera rendering.")
            image_tensor = colormaps.apply_colormap(outputs["accumulation"].detach()).clamp(0.0, 1.0)
        image = (image_tensor * 255.0).to(torch.uint8).cpu().numpy()
        if image.shape[-1] != 3:
            raise ValueError(f"Expected RGB output with 3 channels, got shape {tuple(image.shape)}")
        return image

    def _render_image(
        self,
        camera_state: CameraState,
        phase: RenderPhase,
        render_mode: RenderMode,
        depth_quantile: float,
    ) -> np.ndarray:
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
        return self._convert_output_to_image(outputs, render_mode, depth_quantile)

    def _send_image(self, image: np.ndarray, phase: RenderPhase, render_mode: RenderMode, depth_quantile: float) -> None:
        with self._latest_image_lock:
            self._latest_image = image.copy()
        quality = self.render_config.jpeg_quality_static if phase == "static" else self.render_config.jpeg_quality_moving
        self.client.scene.set_background_image(
            image,
            format=self.image_format,
            jpeg_quality=quality,
        )
        quantile_suffix = f", q={depth_quantile:.2f}" if render_mode == "depth" else ""
        self.on_render_status(f"Rendered {render_mode} {image.shape[1]}x{image.shape[0]} ({phase}{quantile_suffix})")

    def run(self) -> None:
        while self._running:
            if not self._trigger.wait(self.render_config.idle_sleep_seconds):
                if self._request is not None and (time.time() - self._last_motion_time) >= self.render_config.static_transition_seconds:
                    self.submit(
                        RenderRequest(
                            camera_state=self._request.camera_state,
                            phase="static",
                            render_mode=self._request.render_mode,
                            depth_quantile=self._request.depth_quantile,
                        )
                    )
                continue
            self._trigger.clear()
            request = self._request
            if request is None or not self._running:
                continue
            try:
                image = self._render_image(
                    request.camera_state,
                    request.phase,
                    request.render_mode,
                    request.depth_quantile,
                )
                if not self._running:
                    break
                self._send_image(image, request.phase, request.render_mode, request.depth_quantile)
            except Exception as exc:
                self.on_render_status(f"Render error: {type(exc).__name__}")
                if not self._running:
                    break
