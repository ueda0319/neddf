from abc import ABC, abstractmethod

import torch
from numpy import ndarray
from torch import Tensor, nn


class BaseCameraCalib(ABC, nn.Module):
    """Abstract base class for camera calibration.

    Attributes:
        params (np.ndarray[dim, float]): Camera intrisic parameters
            Note that the number of dimensions differs for each calibration model.
    """

    def __init__(
        self,
        calib_param: ndarray,
    ) -> None:
        """Initializer

        This method initialize common attributes.

        Args:
            calib_param (np.ndarray[dim, float]): Camera intrisic parameters

        Note:
            This method is called during initialization.
            Inherited classes must validate that the number of parameters of
            each calibration model and the dimension of `calib_param` match
            before calling the `initializer` of this abstract class.

        """
        super().__init__()
        self.params = nn.Parameter(torch.from_numpy(calib_param).to(torch.float32))

    @property
    def device(self) -> torch.device:
        return self.params.device

    @abstractmethod
    def project_local(self, xyz: Tensor) -> Tensor:
        """Projection

        Project points in camera coordinate to uv position of the camera.

        Args:
            xyz (Tensor[batch_size, 3, float32]):
                Input point positions [x, y, z] in camera coordinate.
                Note that xyz should be already transformed from world to camera.

        Returns:
            Tensor[batch_size, 2, float32]: pixel position [u, v] of each input points
        """
        raise NotImplementedError()

    @abstractmethod
    def unproject_local(self, uv: Tensor) -> Tensor:
        """Unprojection

        Unproject points in camera coordinate to global coordinate.

        Args:
            pos_camera (Tensor[batch_size, 3, float32]):
                input point positions [u, v, d] in camera coordinate.

        Returns:
            Tensor[batch_size, 3, float32]:
                positions in global coordinate [x, y, z] of each input pixels.
        """
        raise NotImplementedError()
