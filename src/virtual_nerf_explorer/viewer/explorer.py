from __future__ import annotations

import multiprocessing as mp
import threading
import time
from dataclasses import dataclass

import viser

from virtual_nerf_explorer.camera import (
    apply_camera_pose,
    camera_state_from_client,
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
        )
        self.render_lock = threading.Lock()
        self.server = viser.ViserServer(host=app_config.host, port=app_config.port)
        self.scene_handles = build_scene(self.server, session, self.state, app_config)
        self.gui_handles = build_gui(
            self.server,
            config_path=session.config_path,
            checkpoint_step=session.checkpoint_step,
            state=self.state,
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

    def _wire_gui(self) -> None:
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
        pose = self.scene_handles.train_camera_poses[idx]
        with event.client.atomic():
            apply_camera_pose(event.client, pose)
        context = self.client_contexts.get(event.client.client_id)
        if context is not None:
            context.worker.submit(RenderRequest(camera_state=camera_state_from_client(event.client), phase="static"))

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
        worker.submit(RenderRequest(camera_state=camera_state_from_client(client), phase="static"))

        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            context = self.client_contexts.get(client.client_id)
            if context is None:
                return
            with client.atomic():
                enforce_minimum_orbit_distance(client, self.app_config.min_orbit_distance)
            context.worker.submit(RenderRequest(camera_state=camera_state_from_client(client), phase="move"))

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
