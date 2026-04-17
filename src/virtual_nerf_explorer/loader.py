from __future__ import annotations

from pathlib import Path
from typing import Iterable

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils.eval_utils import eval_setup

from virtual_nerf_explorer.session import LoadedSession


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _infer_output_root(config_path: Path) -> Path:
    config_dir = config_path.resolve().parent
    if len(config_dir.parents) >= 3:
        return config_dir.parents[2]
    return config_dir


def _resolve_existing_path(raw_path: Path, *, config_path: Path, output_root: Path) -> Path:
    if raw_path.is_absolute():
        return raw_path

    config_dir = config_path.resolve().parent
    candidates = [Path.cwd() / raw_path, output_root / raw_path]
    candidates.extend(base / raw_path for base in (config_dir, *config_dir.parents))
    for candidate in _dedupe_paths(candidates):
        if candidate.exists():
            return candidate.resolve()
    return (output_root / raw_path).resolve()


def _update_loaded_config(config: TrainerConfig, *, config_path: Path) -> TrainerConfig:
    output_root = _infer_output_root(config_path)
    config.output_dir = output_root
    config.load_config = config_path.resolve()

    for obj in (config, config.pipeline.datamanager, config.pipeline.datamanager.dataparser):
        if hasattr(obj, "data") and getattr(obj, "data") is not None:
            setattr(
                obj,
                "data",
                _resolve_existing_path(Path(getattr(obj, "data")), config_path=config_path, output_root=output_root),
            )
    return config


def load_session(config_path: Path) -> LoadedSession:
    config_path = config_path.expanduser().resolve()
    loaded_config, pipeline, checkpoint_path, checkpoint_step = eval_setup(
        config_path,
        test_mode="inference",
        update_config_callback=lambda cfg: _update_loaded_config(cfg, config_path=config_path),
    )
    return LoadedSession(
        config=loaded_config,
        pipeline=pipeline,
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step,
        config_path=config_path,
        datapath=loaded_config.data,
        viewer_config=loaded_config.viewer,
    )
