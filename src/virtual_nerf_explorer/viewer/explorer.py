from __future__ import annotations

import multiprocessing as mp
import threading
import time
from dataclasses import dataclass

import viser

from virtual_nerf_explorer.camera import (
    StoredCameraView,
    apply_stored_view,
    apply_camera_pose,
    camera_state_from_client,
    capture_camera_view,
    enforce_minimum_orbit_distance,
)
from virtual_nerf_explorer.capture import capture_scene_image
from virtual_nerf_explorer.config import AppConfig
from virtual_nerf_explorer.render import ClientRenderWorker, RenderRequest
from virtual_nerf_explorer.session import LoadedSession
from virtual_nerf_explorer.state import ViewerState
from virtual_nerf_explorer.viewer.gui import GuiHandles, build_gui, update_info
from virtual_nerf_explorer.viewer.scene import (
    SceneHandles,
    build_scene,
    set_training_camera_visibility,
    set_world_axes_visibility,
)


@dataclass(slots=True)
class ClientContext:
    worker: ClientRenderWorker


class SceneExplorer:
    def __init__(self, session: LoadedSession, app_config: AppConfig) -> None:
        self.session = session
        self.app_config = app_config
        self.state = ViewerState(
            show_training_cameras=app_config.show_training_cameras,
            show_world_axes=app_config.show_world_axes,
            depth_quantile=app_config.render.depth_quantile,
        )
        self.render_lock = threading.Lock()
        self.server = viser.ViserServer(host=app_config.host, port=app_config.port)
        self.scene_handles = build_scene(self.server, session, self.state, app_config)
        self.saved_views: dict[str, StoredCameraView] = {}
        self.gui_handles = build_gui(
            self.server,
            config_path=session.config_path,
            checkpoint_step=session.checkpoint_step,
            state=self.state,
            train_camera_options=self._train_camera_options(),
        )
        self.client_contexts: dict[int, ClientContext] = {}
        self._wire_gui()
        self._wire_clients()

    def _set_status(self, status: str) -> None:
        self.state.last_render_status = status
        update_info(
            self.gui_handles,
            config_path=self.session.config_path,
            checkpoint_step=self.session.checkpoint_step,
            state=self.state,
        )

    def _train_camera_options(self) -> tuple[str, ...]:
        if not self.scene_handles.train_camera_indices:
            return ("None",)
        return tuple(f"{idx:05d}" for idx in self.scene_handles.train_camera_indices)

    def _refresh_saved_views(self) -> None:
        options = tuple(self.saved_views.keys()) or ("None",)
        self.gui_handles.saved_view.options = options
        if self.gui_handles.saved_view.value not in options:
            self.gui_handles.saved_view.value = options[0]
        self.state.saved_views = len(self.saved_views)
        update_info(
            self.gui_handles,
            config_path=self.session.config_path,
            checkpoint_step=self.session.checkpoint_step,
            state=self.state,
        )

    def _get_target_client(self, event_client: viser.ClientHandle | None) -> viser.ClientHandle | None:
        if event_client is not None:
            return event_client
        if not self.client_contexts:
            return None
        client_id = next(iter(self.client_contexts))
        return self.server.get_clients().get(client_id)

    def _submit_render_for_client(self, client: viser.ClientHandle, phase: str = "static") -> None:
        context = self.client_contexts.get(client.client_id)
        if context is None:
            return
        context.worker.submit(
            RenderRequest(
                camera_state=camera_state_from_client(client),
                phase=phase,
                render_mode=self.state.render_mode,
                depth_quantile=self.state.depth_quantile,
            )
        )

    def _snap_to_train_camera(self, client: viser.ClientHandle, camera_index: int) -> None:
        pose = self.scene_handles.train_camera_poses[camera_index]
        with client.atomic():
            apply_camera_pose(client, pose)
        self._submit_render_for_client(client)

    def _wire_gui(self) -> None:
        @self.gui_handles.render_mode.on_update
        def _(_: viser.GuiEvent) -> None:
            self.state.render_mode = self.gui_handles.render_mode.value
            client = self._get_target_client(None)
            if client is not None:
                self._submit_render_for_client(client)
            self._set_status(f"Render mode set to {self.state.render_mode}")

        @self.gui_handles.depth_quantile.on_update
        def _(_: viser.GuiEvent) -> None:
            self.state.depth_quantile = self.gui_handles.depth_quantile.value
            self.app_config.render.depth_quantile = self.gui_handles.depth_quantile.value
            client = self._get_target_client(None)
            if client is not None and self.state.render_mode == "depth":
                self._submit_render_for_client(client)
            self._set_status(f"Depth cumsum set to {self.state.depth_quantile:.2f}")

        @self.gui_handles.capture_button.on_click
        def _(event: viser.GuiEvent) -> None:
            if event.client is None:
                self._set_status("Capture requested without an active client")
                return
            context = self.client_contexts.get(event.client.client_id)
            if context is None:
                self._set_status("Capture requested without an active render worker")
                return
            image = context.worker.get_latest_image()
            if image is None:
                self._set_status("No rendered image available yet for capture")
                return
            filename = capture_scene_image(
                client=event.client,
                image=image,
                base_name=self.gui_handles.capture_name.value,
            )
            self._set_status(f"Downloaded capture {filename}")

        @self.gui_handles.navigation_actions.on_click
        def _(event: viser.GuiEvent) -> None:
            action = self.gui_handles.navigation_actions.value
            client = self._get_target_client(event.client)
            if client is None:
                self._set_status(f"{action} requested without an active client")
                return
            if action == "Reset":
                if self.scene_handles.train_camera_poses:
                    first_idx = self.scene_handles.train_camera_indices[0]
                    self._snap_to_train_camera(client, first_idx)
                    self._set_status(f"Reset to training camera {first_idx:05d}")
                else:
                    with client.atomic():
                        apply_camera_pose(client, self.scene_handles.overview_pose)
                    self._submit_render_for_client(client)
                    self._set_status("Reset to overview pose")
                return

            if action == "Focus":
                with client.atomic():
                    apply_camera_pose(client, self.scene_handles.overview_pose)
                self._submit_render_for_client(client)
                self._set_status("Focused scene overview")
                return

            if action == "Next":
                if not self.scene_handles.train_camera_indices:
                    self._set_status("No training cameras available for navigation")
                    return
                current_value = self.gui_handles.train_camera.value
                try:
                    current_idx = self.scene_handles.train_camera_indices.index(int(current_value))
                except (ValueError, TypeError):
                    current_idx = -1
                next_idx = self.scene_handles.train_camera_indices[
                    (current_idx + 1) % len(self.scene_handles.train_camera_indices)
                ]
                self.gui_handles.train_camera.value = f"{next_idx:05d}"
                self._snap_to_train_camera(client, next_idx)
                self._set_status(f"Snapped to training camera {next_idx:05d}")
                return

            if action == "Snap":
                try:
                    camera_index = int(self.gui_handles.train_camera.value)
                except ValueError:
                    self._set_status("No valid training camera selected")
                    return
                if camera_index not in self.scene_handles.train_camera_poses:
                    self._set_status("Selected training camera is not available")
                    return
                self._snap_to_train_camera(client, camera_index)
                self._set_status(f"Snapped to training camera {camera_index:05d}")

        @self.gui_handles.saved_view_actions.on_click
        def _(event: viser.GuiEvent) -> None:
            action = self.gui_handles.saved_view_actions.value
            client = self._get_target_client(event.client)

            if action == "Save":
                if client is None:
                    self._set_status("Save view requested without an active client")
                    return
                name = self.gui_handles.view_name.value.strip() or f"view_{len(self.saved_views) + 1:02d}"
                self.saved_views[name] = capture_camera_view(client)
                self._refresh_saved_views()
                self.gui_handles.saved_view.value = name
                self._set_status(f"Saved view {name}")
                return

            if action == "Load":
                if client is None:
                    self._set_status("Load view requested without an active client")
                    return
                name = self.gui_handles.saved_view.value
                view = self.saved_views.get(name)
                if view is None:
                    self._set_status("No saved view selected")
                    return
                with client.atomic():
                    apply_stored_view(client, view)
                self._submit_render_for_client(client)
                self._set_status(f"Loaded view {name}")
                return

            if action == "Delete":
                name = self.gui_handles.saved_view.value
                if name not in self.saved_views:
                    self._set_status("No saved view selected")
                    return
                del self.saved_views[name]
                self._refresh_saved_views()
                self._set_status(f"Deleted view {name}")

        @self.gui_handles.show_training_cameras.on_update
        def _(_: viser.GuiEvent) -> None:
            self.state.show_training_cameras = self.gui_handles.show_training_cameras.value
            set_training_camera_visibility(self.scene_handles, self.state.show_training_cameras)
            self._set_status("Updated training camera visibility")

        @self.gui_handles.show_world_axes.on_update
        def _(_: viser.GuiEvent) -> None:
            self.state.show_world_axes = self.gui_handles.show_world_axes.value
            set_world_axes_visibility(self.scene_handles, self.state.show_world_axes)
            self._set_status("Updated world visibility")

        @self.gui_handles.render_resolution.on_update
        def _(_: viser.GuiEvent) -> None:
            self.app_config.render.static_max_res = self.gui_handles.render_resolution.value
            self.app_config.render.moving_max_res = max(192, self.gui_handles.render_resolution.value // 3)
            self._set_status(f"Static max resolution set to {self.app_config.render.static_max_res}")

    def _on_frustum_click(self, idx: int, event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
        self.gui_handles.train_camera.value = f"{idx:05d}"
        self._snap_to_train_camera(event.client, idx)

    def _wire_clients(self) -> None:
        for idx, handle in self.scene_handles.train_camera_handles.items():
            handle.on_click(lambda event, idx=idx: self._on_frustum_click(idx, event))

        self.server.on_client_connect(self._handle_client_connect)
        self.server.on_client_disconnect(self._handle_client_disconnect)

    def _handle_client_connect(self, client: viser.ClientHandle) -> None:
        worker = ClientRenderWorker(
            client=client,
            pipeline=self.session.pipeline,
            render_lock=self.render_lock,
            render_config=self.app_config.render,
            image_format=self.session.viewer_config.image_format,
            on_render_status=self._set_status,
        )
        self.client_contexts[client.client_id] = ClientContext(worker=worker)
        self.state.connected_clients = len(self.client_contexts)
        update_info(
            self.gui_handles,
            config_path=self.session.config_path,
            checkpoint_step=self.session.checkpoint_step,
            state=self.state,
        )

        if self.scene_handles.train_camera_poses:
            first_idx = min(self.scene_handles.train_camera_poses)
            apply_camera_pose(client, self.scene_handles.train_camera_poses[first_idx])
        worker.start()
        worker.submit(
            RenderRequest(
                camera_state=camera_state_from_client(client),
                phase="static",
                render_mode=self.state.render_mode,
                depth_quantile=self.state.depth_quantile,
            )
        )

        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            context = self.client_contexts.get(client.client_id)
            if context is None:
                return
            with client.atomic():
                enforce_minimum_orbit_distance(client, self.app_config.min_orbit_distance)
            context.worker.submit(
                RenderRequest(
                    camera_state=camera_state_from_client(client),
                    phase="move",
                    render_mode=self.state.render_mode,
                    depth_quantile=self.state.depth_quantile,
                )
            )

    def _handle_client_disconnect(self, client: viser.ClientHandle) -> None:
        context = self.client_contexts.pop(client.client_id, None)
        if context is not None:
            context.worker.stop()
            context.worker.join(timeout=2.0)
        self.state.connected_clients = len(self.client_contexts)
        self._set_status("Waiting for client" if not self.client_contexts else "Client disconnected")

    def _shutdown_background_processes(self) -> None:
        children = list(mp.active_children())
        for child in children:
            child.terminate()
        for child in children:
            child.join(timeout=1.0)
        for child in children:
            if child.is_alive():
                child.kill()
                child.join(timeout=1.0)

    def shutdown(self) -> None:
        contexts = list(self.client_contexts.values())
        for context in contexts:
            context.worker.stop()
        self.client_contexts.clear()
        self.server.stop()
        for context in contexts:
            context.worker.join(timeout=2.0)
        self._shutdown_background_processes()

    def run_forever(self) -> None:
        self._set_status("Waiting for client")
        print(f"{self.app_config.title} running at http://{self.server.get_host()}:{self.server.get_port()}")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.shutdown()
