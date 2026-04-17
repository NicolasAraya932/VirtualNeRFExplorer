from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import viser
import viser.transforms as vtf
from nerfstudio.cameras.cameras import Cameras

from virtual_nerf_explorer.camera import VISER_NERFSTUDIO_SCALE_RATIO
from virtual_nerf_explorer.config import AppConfig
from virtual_nerf_explorer.session import LoadedSession
from virtual_nerf_explorer.state import ViewerState


@dataclass(slots=True)
class SceneHandles:
    grid: viser.GridHandle
    origin_frame: viser.FrameHandle
    train_camera_handles: dict[int, viser.CameraFrustumHandle]
    train_camera_poses: dict[int, np.ndarray]
    train_camera_indices: list[int]
    scene_center: np.ndarray
    overview_pose: np.ndarray


def _pick_indices(total: int, limit: int) -> list[int]:
    if total <= 0:
        return []
    limit = max(1, min(total, limit))
    return np.linspace(0, total - 1, num=limit, dtype=np.int32).tolist()


def _camera_fov(camera: Cameras) -> float:
    return float(2.0 * np.arctan((camera.cx / camera.fx[0]).cpu().item()))


def _look_at_pose(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - position
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-6:
        forward = np.array([0.0, 1.0, -0.2], dtype=np.float64)
        forward_norm = float(np.linalg.norm(forward))
    forward = forward / forward_norm
    up = up / max(float(np.linalg.norm(up)), 1e-6)
    right = np.cross(up, forward)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / right_norm
    true_up = np.cross(forward, right)
    true_up = true_up / max(float(np.linalg.norm(true_up)), 1e-6)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = forward
    pose[:3, 3] = position
    return pose[:3, :]


def build_scene(
    server: viser.ViserServer,
    session: LoadedSession,
    state: ViewerState,
    app_config: AppConfig,
) -> SceneHandles:
    server.scene.set_up_direction("+z")
    grid = server.scene.add_grid(
        "/world/grid",
        width=4.0,
        height=4.0,
        width_segments=20,
        height_segments=20,
        visible=state.show_world_axes,
    )
    origin_frame = server.scene.add_frame(
        "/world/origin",
        axes_length=0.4,
        axes_radius=0.015,
        visible=state.show_world_axes,
    )

    train_camera_handles: dict[int, viser.CameraFrustumHandle] = {}
    train_camera_poses: dict[int, np.ndarray] = {}
    train_camera_indices: list[int] = []
    cameras = session.pipeline.datamanager.train_dataparser_outputs.cameras
    all_positions = cameras.camera_to_worlds[:, :3, 3].cpu().numpy()
    if len(all_positions) > 0:
        scene_center = all_positions.mean(axis=0)
        radius = float(np.linalg.norm(all_positions - scene_center[None, :], axis=1).max())
    else:
        scene_center = np.zeros(3, dtype=np.float64)
        radius = 1.0
    overview_position = scene_center + np.array([radius * 1.6, -radius * 1.6, radius * 0.9], dtype=np.float64)
    overview_pose = _look_at_pose(overview_position, scene_center, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    for idx in _pick_indices(len(cameras), app_config.max_display_cameras):
        camera = cameras[idx]
        c2w = camera.camera_to_worlds.cpu().numpy()
        rotation = vtf.SO3.from_matrix(c2w[:3, :3]) @ vtf.SO3.from_x_radians(np.pi)
        handle = server.scene.add_camera_frustum(
            name=f"/train_cameras/{idx:05d}",
            fov=_camera_fov(camera),
            aspect=float((camera.cx[0] / camera.cy[0]).cpu().item()),
            scale=session.viewer_config.camera_frustum_scale,
            color=(255, 170, 0),
            wxyz=rotation.wxyz,
            position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
            visible=state.show_training_cameras,
        )
        train_camera_handles[idx] = handle
        train_camera_poses[idx] = c2w
        train_camera_indices.append(idx)
    return SceneHandles(
        grid=grid,
        origin_frame=origin_frame,
        train_camera_handles=train_camera_handles,
        train_camera_poses=train_camera_poses,
        train_camera_indices=train_camera_indices,
        scene_center=scene_center,
        overview_pose=overview_pose,
    )


def set_training_camera_visibility(scene_handles: SceneHandles, visible: bool) -> None:
    for handle in scene_handles.train_camera_handles.values():
        handle.visible = visible


def set_world_axes_visibility(scene_handles: SceneHandles, visible: bool) -> None:
    scene_handles.grid.visible = visible
    scene_handles.origin_frame.visible = visible
