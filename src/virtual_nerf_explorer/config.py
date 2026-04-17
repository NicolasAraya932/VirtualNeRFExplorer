from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RenderConfig:
    static_max_res: int = 960
    moving_max_res: int = 320
    depth_quantile: float = 0.5
    jpeg_quality_static: int = 90
    jpeg_quality_moving: int = 60
    idle_sleep_seconds: float = 0.2
    static_transition_seconds: float = 0.15


@dataclass(slots=True)
class AppConfig:
    load_config: Path
    host: str = "0.0.0.0"
    port: int = 8080
    title: str = "VirtualNeRFExplorer"
    show_training_cameras: bool = True
    show_world_axes: bool = True
    max_display_cameras: int = 24
    min_orbit_distance: float = 1.0
    render: RenderConfig = field(default_factory=RenderConfig)
