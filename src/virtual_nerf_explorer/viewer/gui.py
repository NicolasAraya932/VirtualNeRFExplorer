from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import viser
import viser.theme

from virtual_nerf_explorer.state import ViewerState


@dataclass(slots=True)
class GuiHandles:
    info_markdown: viser.GuiMarkdownHandle
    show_training_cameras: viser.GuiInputHandle[bool]
    show_world_axes: viser.GuiInputHandle[bool]
    render_resolution: viser.GuiInputHandle[int]


def _info_text(*, config_path: Path, checkpoint_step: int, state: ViewerState) -> str:
    return "\n".join(
        [
            "# Scene Explorer",
            f"- Config: `{config_path}`",
            f"- Checkpoint step: `{checkpoint_step}`",
            f"- Connected clients: `{state.connected_clients}`",
            f"- Status: `{state.last_render_status}`",
        ]
    )


def build_gui(
    server: viser.ViserServer,
    *,
    config_path: Path,
    checkpoint_step: int,
    state: ViewerState,
) -> GuiHandles:
    server.gui.configure_theme(
        control_layout="collapsible",
        control_width="large",
        dark_mode=True,
        show_logo=False,
        brand_color=(255, 170, 0),
    )
    server.gui.set_panel_label("Explorer")

    info_markdown = server.gui.add_markdown(_info_text(config_path=config_path, checkpoint_step=checkpoint_step, state=state))
    with server.gui.add_folder("Scene"):
        show_training_cameras = server.gui.add_checkbox(
            "Show training cameras",
            initial_value=state.show_training_cameras,
        )
        show_world_axes = server.gui.add_checkbox(
            "Show world grid/frame",
            initial_value=state.show_world_axes,
        )
        render_resolution = server.gui.add_slider(
            "Static max resolution",
            min=256,
            max=1600,
            step=64,
            initial_value=960,
        )
    return GuiHandles(
        info_markdown=info_markdown,
        show_training_cameras=show_training_cameras,
        show_world_axes=show_world_axes,
        render_resolution=render_resolution,
    )


def update_info(gui: GuiHandles, *, config_path: Path, checkpoint_step: int, state: ViewerState) -> None:
    gui.info_markdown.content = _info_text(config_path=config_path, checkpoint_step=checkpoint_step, state=state)
