from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
from neddf.ray import Sampling
from numpy import ndarray
from torch import Tensor, nn


class BaseNeuralField(ABC, nn.Module):
    """Abstract base class for NeuralField."""

    def set_iter(self, iter: int) -> None:
        """Set iteration

        This methods set iteration number for configure warm ups

        Args:
            iter (int): current iteration. Set -1 for evaluation.
        """
        pass

    @abstractmethod
    def forward(
        self,
        sampling: Sampling,
    ) -> Dict[str, Tensor]:
        """Forward propagation

        This methods should return density and color at minimum

        Args:
            sampling (Sampling[batch_size, sampling, 3])
        """
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        """Forward propagation

        This methods should return density and color at minimum

        Returns:
            torch.device: device information of this module
        """
        return next(self.parameters()).device

    def voxelize(
        self,
        field_name: str = "density",
        cube_range: float = 1.1,
        cube_resolution: int = 64,
        chunk: int = 65536,
    ) -> ndarray:
        with torch.set_grad_enabled(False):
            ids: ndarray = np.linspace(-cube_range, cube_range, cube_resolution)
            zs_np, ys_np, xs_np = np.meshgrid(ids, ids, ids)
            xs: Tensor = torch.from_numpy(xs_np.astype(np.float32)).view(-1)
            ys: Tensor = torch.from_numpy(ys_np.astype(np.float32)).view(-1)
            zs: Tensor = torch.from_numpy(zs_np.astype(np.float32)).view(-1)
            sampling_count: int = cube_resolution ** 3
            sample_pos = torch.stack([xs, ys, zs], 1)
            sample_dir = torch.tensor([[1.0, 0.0, 0.0]]).expand(sampling_count, -1)
            diag_variance = torch.tensor([[0.0, 0.0, 0.0]]).expand(sampling_count, -1)
            device: torch.device = self.device

            result: ndarray = np.zeros(sampling_count, np.float32)
            for i in range(0, sampling_count, chunk):
                i_next: int = min(sampling_count, i + chunk)

                sampling: Sampling = Sampling(
                    sample_pos=sample_pos[None, i:i_next, :].to(device),
                    sample_dir=sample_dir[None, i:i_next, :].to(device),
                    diag_variance=diag_variance[None, i:i_next, :].to(device),
                )
                val: Dict[str, Tensor] = self.forward(sampling)
                result[i:i_next] = val[field_name].view(-1).detach().cpu().numpy()
            return result.reshape(cube_resolution, cube_resolution, cube_resolution)
