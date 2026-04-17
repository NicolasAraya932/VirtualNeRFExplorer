from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ViewerState:
    show_training_cameras: bool = True
    show_world_axes: bool = True
    connected_clients: int = 0
    last_render_status: str = "Waiting for client"
