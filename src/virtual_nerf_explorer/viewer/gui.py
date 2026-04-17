from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import viser
import viser.theme

from virtual_nerf_explorer.state import ViewerState


@dataclass(slots=True)
class GuiHandles:
    info_markdown: viser.GuiMarkdownHandle
    render_mode: viser.GuiInputHandle[str]
    depth_quantile: viser.GuiInputHandle[float]
    capture_name: viser.GuiInputHandle[str]
    capture_button: viser.GuiButtonHandle
    train_camera: viser.GuiInputHandle[str]
    navigation_actions: viser.GuiInputHandle[str]
    view_name: viser.GuiInputHandle[str]
    saved_view: viser.GuiInputHandle[str]
    saved_view_actions: viser.GuiInputHandle[str]
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
            f"- Render mode: `{state.render_mode}`",
            f"- Depth cumsum: `{state.depth_quantile:.2f}`",
            f"- Saved views: `{state.saved_views}`",
            f"- Status: `{state.last_render_status}`",
        ]
    )


def build_gui(
    server: viser.ViserServer,
    *,
    config_path: Path,
    checkpoint_step: int,
    state: ViewerState,
    train_camera_options: tuple[str, ...],
) -> GuiHandles:
    server.gui.configure_theme(
        control_layout="collapsible",
        dark_mode=True,
        show_logo=False,
        brand_color=(255, 170, 0),
    )
    server.gui.set_panel_label("Explorer")

    info_markdown = server.gui.add_markdown(
        _info_text(config_path=config_path, checkpoint_step=checkpoint_step, state=state)
    )
    tab_group = server.gui.add_tab_group()

    with tab_group.add_tab("Scene", icon=viser.Icon.CUBE):
        render_mode = server.gui.add_dropdown(
            "Render",
            options=("rgb", "depth", "accumulation"),
            initial_value=state.render_mode,
            hint="Choose the rendered output shown in the viewport.",
        )
        depth_quantile = server.gui.add_slider(
            "Depth cumsum",
            min=0.05,
            max=0.95,
            step=0.01,
            initial_value=state.depth_quantile,
        )
        capture_name = server.gui.add_text(
            "Name",
            initial_value="scene_capture",
            hint="Base filename for downloads.",
        )
        capture_button = server.gui.add_button(
            "Capture",
            icon=viser.Icon.CAMERA,
            hint="Download the current client view using the chosen base name.",
        )
        with server.gui.add_folder("Navigate"):
            train_camera = server.gui.add_dropdown(
                "Camera",
                options=train_camera_options,
                initial_value=train_camera_options[0],
                hint="Displayed training camera to snap to.",
            )
            navigation_actions = server.gui.add_button_group(
                "Actions",
                ("Reset", "Focus", "Next", "Snap"),
                hint="Scene navigation controls.",
            )

    with tab_group.add_tab("Views", icon=viser.Icon.BOOKMARK):
        view_name = server.gui.add_text(
            "Name",
            initial_value="view_01",
            hint="Name for saving the current camera pose.",
        )
        saved_view = server.gui.add_dropdown(
            "Saved",
            options=("None",),
            initial_value="None",
            hint="Choose a saved camera pose.",
        )
        saved_view_actions = server.gui.add_button_group(
            "Actions",
            ("Save", "Load", "Delete"),
            hint="Save, restore, or remove stored viewpoints.",
        )

    with tab_group.add_tab("Display", icon=viser.Icon.SETTINGS):
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
        render_mode=render_mode,
        depth_quantile=depth_quantile,
        capture_name=capture_name,
        capture_button=capture_button,
        train_camera=train_camera,
        navigation_actions=navigation_actions,
        view_name=view_name,
        saved_view=saved_view,
        saved_view_actions=saved_view_actions,
        show_training_cameras=show_training_cameras,
        show_world_axes=show_world_axes,
        render_resolution=render_resolution,
    )


def update_info(gui: GuiHandles, *, config_path: Path, checkpoint_step: int, state: ViewerState) -> None:
    gui.info_markdown.content = _info_text(config_path=config_path, checkpoint_step=checkpoint_step, state=state)
