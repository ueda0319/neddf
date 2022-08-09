import torch
from neddf.camera.base_camera_calib import BaseCameraCalib
from numpy import ndarray
from torch import Tensor, nn


class PinholeCalib(BaseCameraCalib):
    def __init__(
        self,
        calib_param: ndarray,
    ) -> None:
        """Initializer

        This method initializes the pinhole camera model [fx, fy, cx, cy].
        It validates that the dimension number of the argument is suitable,
        and then calls the superclass initialization.

        Args:
            calib_param (ndarray[4, float]): Camera intrisic parameters [fx, fy, cx, cy]
        """
        assert calib_param.shape == (4,)
        super().__init__(calib_param)

    def project_local(self, xyz: Tensor) -> Tensor:
        """Projection

        Project points in camera coordinate to uv position of the camera

        Args:
            xyz (Tensor[batch_size, 3, float32]):
                Input point positions [x, y, z] in camera coordinate
                Note that xyz should be already transformed from world to camera

        Returns:
            Tensor[batch_size, 2, float32]: pixel position [u, v] of each input points
        """
        # 3d position in camera coordinate describe position xyz in Right-Up-Back
        # To projection, convert it to Right-Down-Front
        rub2rdf: Tensor = torch.eye(3, dtype=torch.float32, device=self.device)
        rub2rdf[1, 1] = -1
        rub2rdf[2, 2] = -1
        xyz_rdf = torch.matmul(rub2rdf, xyz[:, :, None])[:, :, 0]

        # inverse z
        zi: Tensor = torch.reciprocal(xyz_rdf[:, 2])
        u: Tensor = self.fx * xyz_rdf[:, 0] * zi + self.cx
        v: Tensor = self.fy * xyz_rdf[:, 1] * zi + self.cy
        uv = torch.stack([u, v], 1)
        return uv

    def unproject_local(self, uv: Tensor) -> Tensor:
        """Unprojection

        Unproject points in camera coordinate to global coordinate

        Args:
            pos_camera (Tensor[batch_size, 3, float32]):
                input point positions [u, v, d] in camera coordinate

        Returns:
            Tensor[batch_size, 3, float32]:
                positions in global coordinate [x, y, z] of each input pixels
        """
        x: Tensor = (1.0 / self.fx) * (uv[:, 0] - self.cx)
        y: Tensor = (1.0 / self.fy) * (uv[:, 1] - self.cy)
        z: Tensor = torch.ones_like(x)
        xyz_rdf = torch.stack([x, y, z], 0)
        # Convert axis Right-Down-Front to Right-Up-Back
        rdf2rub: Tensor = torch.eye(3, dtype=torch.float32, device=self.device)
        rdf2rub[1, 1] = -1
        rdf2rub[2, 2] = -1
        # Apply RUB and normalize
        xyz_rub = nn.functional.normalize(torch.matmul(rdf2rub, xyz_rdf).T, p=2, dim=1)
        return xyz_rub

    @property
    def fx(self) -> Tensor:
        """Tensor[1, float]: horizontal focal length"""
        return self.params[0]

    @property
    def fy(self) -> Tensor:
        """Tensor[1, float]: vertical focal length"""
        return self.params[1]

    @property
    def cx(self) -> Tensor:
        """Tensor[1, float]: horizontal camera center position"""
        return self.params[2]

    @property
    def cy(self) -> Tensor:
        """Tensor[1, float]: vertical camera center position"""
        return self.params[3]
