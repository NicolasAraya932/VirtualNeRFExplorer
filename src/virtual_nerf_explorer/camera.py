from __future__ import annotations

import numpy as np
import torch
import viser
import viser.transforms as vtf
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.viewer.utils import CameraState

VISER_NERFSTUDIO_SCALE_RATIO = 10.0


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
    client.camera.position = c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
    client.camera.wxyz = rotation.wxyz
