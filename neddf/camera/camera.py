from typing import Final, Optional

import numpy as np
import torch
from neddf.camera.base_camera_calib import BaseCameraCalib
from neddf.camera.ray import Ray
from numpy import ndarray
from scipy.spatial.transform import Rotation
from torch import Tensor, nn


class Camera(nn.Module):
    def __init__(
        self,
        camera_calib: BaseCameraCalib,
        initial_camera_param: Optional[ndarray] = None,
    ) -> None:
        """Initializer

        This method initialize common attributes.
        This method sets the calibration model and initial camera pose
        parameters from the arguments.
        Initializer generates the SE(3) parameters for adjust camera pose
        and computes the homogeneous transformation matrix [R,T] with the initial pose.

        Args:
            camera_calib (BaseCameraCalib):
                Camera calibration instance
            initial_camera_param (np.ndarray[6, float]):
                SE(3) parameter of initial camera pose.
                If not specified, zero vector is used.
        """
        super().__init__()
        if initial_camera_param is None:
            # initialize with zero vector as default value
            initial_camera_param = np.zeros(6, dtype=np.float32)

        self.camera_calib: BaseCameraCalib = camera_calib
        self.initial_params_np: ndarray = initial_camera_param
        self.params: Tensor = nn.Parameter(torch.zeros(6, dtype=torch.float32))
        self.R: Tensor = torch.eye(3, dtype=torch.float32)
        self.T: Tensor = torch.zeros(3, dtype=torch.float32)
        self.update_transform()

    @property
    def device(self) -> torch.device:
        return self.params.device

    @property
    def R0(self) -> Tensor:
        return torch.from_numpy(
            Rotation.from_rotvec(self.initial_params_np[:3])
            .as_matrix()
            .astype(np.float32)
        ).to(self.device)

    @property
    def T0(self) -> Tensor:
        return torch.from_numpy(self.initial_params_np[3:6].astype(np.float32)).to(
            self.device
        )

    # Calc Transform from camera parameters with gradients
    def update_transform(self) -> None:
        """Update transform

        Update transform of the camera from parameters
        Please call this method after update self.params
        """
        # Calculate rotation matrix from Rodrigues
        i: Tensor = torch.eye(
            3, dtype=torch.float32, device=self.device, requires_grad=False
        )
        w1: Tensor = torch.zeros(
            3, 3, dtype=torch.float32, device=self.device, requires_grad=False
        )
        w2: Tensor = torch.zeros(
            3, 3, dtype=torch.float32, device=self.device, requires_grad=False
        )
        w3: Tensor = torch.zeros(
            3, 3, dtype=torch.float32, device=self.device, requires_grad=False
        )
        w1[1, 2] = -1
        w1[2, 1] = 1
        w2[2, 0] = -1
        w2[0, 2] = 1
        w3[0, 1] = -1
        w3[1, 0] = 1

        theta: Tensor = torch.norm(self.params[0:3])  # type: ignore
        if theta > 1e-10:
            theta_inv: Tensor = 1.0 / theta
            # 3d rotation axis
            n: Tensor = theta_inv * self.params[0:3]
            # cos and sin of theta
            # (they are just 1d float but described in Tensor for keep gradients.)
            c: Tensor = torch.cos(theta)
            s: Tensor = torch.sin(theta)
            w: Tensor = n[0] * w1 + n[1] * w2 + n[2] * w3
            ww: Tensor = torch.matmul(w, w)

            Ri: Tensor = i + s * w + (1.0 - c) * ww
            Vi: Tensor = (
                i
                + (1 - c) * theta_inv * theta_inv * w
                + (theta - s) * theta_inv * theta_inv * theta_inv * ww
            )
        else:
            w = self.params[0] * w1 + self.params[1] * w2 + self.params[2] * w3
            Ri = i + w
            Vi = Ri
        self.R = torch.matmul(Ri, self.R0)
        self.T = (
            torch.matmul(Vi, self.params[3:6, None])
            + torch.matmul(Ri, self.T0[:, None])
        )[:, 0]

    def project(self, pos_world: Tensor) -> Tensor:
        """Projection

        Project points in world coordinate to pixel position of the camera

        Args:
            pos_world (Tensor[batch_size, 3, float32]):
                input point positions [x, y, z] in world coordinate

        Returns:
            Tensor[batch_size, 2, float32]: pixel position [u, v] of each input points
        """
        pos_camera: Tensor = torch.matmul(
            self.R.T, (pos_world[:, :, None] - self.T[None, :, None])
        )[:, :, 0]
        uv: Tensor = self.camera_calib.project_local(pos_camera)
        return uv

    def unproject(self, uv: Tensor) -> Tensor:
        """Unprojection

        Unproject points from uv pixel coordinate to global coordinate

        Args:
            uv (Tensor[batch_size, 2, float32]):
                input point positions [u, v] in pixel coordinate

        Returns:
            Tensor[batch_size, 3, float32]:
                positions in global coordinate [x, y, z] of each input pixels
        """
        pos_camera: Tensor = self.camera_calib.unproject_local(uv)
        pos_world: Tensor = torch.matmul(self.R, pos_camera.T).T + self.T[None, :]
        return pos_world

    def create_rays(self, uv: Tensor) -> Ray:
        """create rays

        Create Ray's instances from target positions in pixel coordinate

        Args:
            uv (Tensor[batch_size, int16]): pixel 2d indexes of target ray

        Returns:
            Ray: rays with length [batch_size]
        """
        batch_size: Final[int] = uv.shape[0]
        uv_center = self.get_center_of_pixels(uv)
        pos_camera: Tensor = self.camera_calib.unproject_local(uv_center)
        ray_dir: Tensor = torch.matmul(self.R, pos_camera.T).T
        ray_orig: Tensor = self.T[None, :].expand(batch_size, 3)
        return Ray(ray_dir, ray_orig, uv)

    def get_center_of_pixels(self, pixel_id: Tensor, scale: float = 1.0) -> Tensor:
        """convert pixel_id to pos

        Convert pixel id to position of camera uv coordinate

        Args:
            pixel_id (Tensor[batch_size, 2, int16]): pixel 2D indexes described like [u, v]

        Returns:
            Tuple:
                Tensor[batch_size, 2, float32]: center position of target pixels
                                                in image coordinate (h, w)
        """
        uv = 0.5 + scale * pixel_id.to(torch.float32)
        return uv
