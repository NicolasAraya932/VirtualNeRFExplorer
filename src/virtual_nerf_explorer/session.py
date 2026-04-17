from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline


@dataclass(slots=True)
class LoadedSession:
    config: TrainerConfig
    pipeline: Pipeline
    checkpoint_path: Path
    checkpoint_step: int
    config_path: Path
    datapath: Path
    viewer_config: ViewerConfig
