from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import viser
import viser.transforms as vtf
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.viewer.utils import CameraState

VISER_NERFSTUDIO_SCALE_RATIO = 10.0


@dataclass(slots=True)
class StoredCameraView:
    position: np.ndarray
    look_at: np.ndarray
    up_direction: np.ndarray
    wxyz: np.ndarray
    fov: float


def camera_state_from_client(client: viser.ClientHandle) -> CameraState:
    rotation = vtf.SO3(wxyz=client.camera.wxyz)
    rotation = rotation @ vtf.SO3.from_x_radians(np.pi)
    rotation_matrix = torch.tensor(rotation.as_matrix(), dtype=torch.float32)
    position = torch.tensor(client.camera.position, dtype=torch.float32) / VISER_NERFSTUDIO_SCALE_RATIO
    c2w = torch.concatenate([rotation_matrix, position[:, None]], dim=1)
    aspect = float(client.camera.aspect) if float(client.camera.aspect) > 0.0 else 1.0
    fov = float(client.camera.fov) if float(client.camera.fov) > 1e-4 else np.deg2rad(60.0)
    return CameraState(
        fov=fov,
        aspect=aspect,
        c2w=c2w,
        camera_type=CameraType.PERSPECTIVE,
    )


def apply_camera_pose(client: viser.ClientHandle, c2w: np.ndarray) -> None:
    rotation = vtf.SO3.from_matrix(c2w[:3, :3])
    rotation = rotation @ vtf.SO3.from_x_radians(np.pi)
    position = c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
    forward = rotation.as_matrix()[:, 2]
    client.camera.position = position
    client.camera.look_at = position + forward * max(1.0, np.linalg.norm(position) * 0.25)
    client.camera.up_direction = rotation.as_matrix()[:, 1]
    client.camera.wxyz = rotation.wxyz


def capture_camera_view(client: viser.ClientHandle) -> StoredCameraView:
    return StoredCameraView(
        position=np.asarray(client.camera.position, dtype=np.float64),
        look_at=np.asarray(client.camera.look_at, dtype=np.float64),
        up_direction=np.asarray(client.camera.up_direction, dtype=np.float64),
        wxyz=np.asarray(client.camera.wxyz, dtype=np.float64),
        fov=float(client.camera.fov),
    )


def apply_stored_view(client: viser.ClientHandle, view: StoredCameraView) -> None:
    client.camera.position = view.position
    client.camera.look_at = view.look_at
    client.camera.up_direction = view.up_direction
    client.camera.wxyz = view.wxyz
    client.camera.fov = view.fov


def enforce_minimum_orbit_distance(client: viser.ClientHandle, minimum_distance: float) -> bool:
    if minimum_distance <= 0.0:
        return False

    position = np.asarray(client.camera.position, dtype=np.float64)
    look_at = np.asarray(client.camera.look_at, dtype=np.float64)
    offset = look_at - position
    distance = float(np.linalg.norm(offset))
    if distance >= minimum_distance:
        return False

    if distance > 1e-6:
        direction = offset / distance
    else:
        rotation = vtf.SO3(wxyz=client.camera.wxyz).as_matrix()
        direction = rotation[:, 2]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            direction = direction / direction_norm

    client.camera.look_at = position + direction * minimum_distance
    return True
