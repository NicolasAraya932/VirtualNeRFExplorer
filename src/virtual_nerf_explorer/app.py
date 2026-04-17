from __future__ import annotations

from virtual_nerf_explorer.config import AppConfig
from virtual_nerf_explorer.loader import load_session
from virtual_nerf_explorer.viewer.explorer import SceneExplorer


def run_app(config: AppConfig) -> None:
    session = load_session(config.load_config)
    explorer = SceneExplorer(session, config)
    explorer.run_forever()
