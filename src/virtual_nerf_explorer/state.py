from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ViewerState:
    show_training_cameras: bool = True
    show_world_axes: bool = True
    render_mode: str = "rgb"
    depth_quantile: float = 0.5
    connected_clients: int = 0
    saved_views: int = 0
    last_render_status: str = "Waiting for client"
