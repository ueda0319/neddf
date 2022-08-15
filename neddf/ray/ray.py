from typing import Final, Tuple

import torch
from neddf.ray.sampling import Sampling
from torch import Tensor


class Ray:
    """Ray

    This class hold rays information which can make up samples

    Attributes:
        single (bool): Batchidied or not.
        ray_dir (Tensor[batch_size, 3, float] or Tensor[3, float]):
            Ray Direction vectors.
        ray_orig (Tensor[batch_size, 3, float] or Tensor[3, float]):
            Ray origin(camera position).
        uv (Tensor[batch_size, 2, float] or Tensor[2, float]):
            Position in 2D pixel coordinate.
    """

    def __init__(
        self,
        ray_dir: Tensor,
        ray_orig: Tensor,
        uv: Tensor,
    ) -> None:
        """Initializer

        This method initialize LinearGradLayer module.

        Args:
            ray_dir (Tensor[batch_size, 3, float] or Tensor[3, float]):
                Ray Direction vectors.
            ray_orig (Tensor[batch_size, 3, float] or Tensor[3, float]):
                Ray origin(camera position).
            uv (Tensor[batch_size, 2, float] or Tensor[2, float]):
                Position in 2D pixel coordinate.

        """
        self.single: Final[bool] = ray_dir.dim() == 1
        # validation for data shapes
        assert ray_orig.shape == ray_dir.shape
        if self.single:
            assert uv.shape == (2,)
        else:
            assert uv.shape == (ray_orig.shape[0], 2)

        self.ray_dir: Tensor = ray_dir
        self.ray_orig: Tensor = ray_orig
        self.uv: Tensor = uv

    @property
    def device(self) -> torch.device:
        """torch.device: device information(ex: cpu, cuda:0) of this instance"""
        return self.ray_dir.device

    def __len__(self) -> int:
        """Special method called in len(self)

        Get count of rays(batchsize)
        """
        if self.single:
            return 1

        return self.ray_dir.shape[0]

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        """Special method called in self[item]

        Get item in selected index

        Args:
            item (int): index of item

        Returns:
            Tuple[
                Tensor[3, float]: Ray direction
                Tensor[3, float]: Ray origin position
            ]
        """
        if self.single:
            return (self.ray_dir, self.ray_orig)

        return (self.ray_dir[item, :], self.ray_orig[item, :])

    def get_sampling_points(
        self,
        dists: Tensor,
    ) -> Sampling:
        """getSamplingPoints

        Obtain 3d positions on the ray calculated from the distances of
        each sampling points from the camera(ray_orig).

        Args:
            dists (Tensor[batch_size, sample_count, float]):
                distance of each sampling points from the ray_orig

        Returns:
            Dict[str, Tensor]:
                <sample_pos> Tensor[batch_size, sample_count, 3, float]:
                    3d positions of each sampling points.
                <sample_dir> Tensor[batch_size, sample_count, 3, float]:
                    3d direction vector of each sampling points.
                    vectors are normalized to norm 1.0.
        """
        batch_size: int = dists.shape[0]
        sample_count: int = dists.shape[1]
        assert batch_size == self.ray_dir.shape[0]

        sample_orig: Tensor = self.ray_orig.unsqueeze(1).expand(
            batch_size, sample_count, 3
        )
        sample_dir: Tensor = self.ray_dir.unsqueeze(1).expand(
            batch_size, sample_count, 3
        )
        sample_pos: Tensor = sample_orig + sample_dir * dists.unsqueeze(2)
        diag_variance: Tensor = torch.zeros_like(sample_orig)
        sampling: Sampling = Sampling(
            sample_pos,
            sample_dir,
            diag_variance,
        )
        return sampling

    def get_sampling_cones(
        self,
        dists: Tensor,
        ray_radius: float = 1e-3,
    ) -> Sampling:
        """get SamplingCones

        Obtain 3d positions on the ray calculated from the distances of
        each sampling points from the camera(ray_orig).

        Args:
            dists (Tensor[batch_size, sample_count, float]):
                distance of each sampling points from the ray_orig

        Returns:
            Dict[str, Tensor]:
                <sample_pos> Tensor[batch_size, sample_count, 3, float]:
                    3d positions of each sampling points.
                <sample_dir> Tensor[batch_size, sample_count, 3, float]:
                    3d direction vector of each sampling points.
                    vectors are normalized to norm 1.0.
        """
        batch_size: int = dists.shape[0]
        sample_count: int = dists.shape[1]
        assert batch_size == self.ray_dir.shape[0]

        sample_orig: Tensor = (
            self.ray_orig.unsqueeze(1).expand(batch_size, sample_count, 3).contiguous()
        )
        sample_dir: Tensor = (
            self.ray_dir.unsqueeze(1).expand(batch_size, sample_count, 3).contiguous()
        )
        dists_near = dists
        dists_far = torch.cat(
            [dists[:, 1:], 2 * dists[:, -1:] - dists[:, -2:-1]], dim=1
        )
        d_mu = 0.5 * (dists_near + dists_far)
        d_sigma = 0.5 * (dists_far - dists_near)
        d_mu2 = torch.square(d_mu)
        d_sigma2 = torch.square(d_sigma)
        d_sigma4 = torch.square(d_sigma2)

        m_inv = torch.reciprocal(3 * d_mu2 + d_sigma2 + 1e-7)
        t_mu = d_mu + (2 * d_mu * d_sigma2) * m_inv
        t_var = (1.0 / 3) * d_sigma2 - (4.0 / 15) * d_sigma4 * (
            12 * d_mu2 - d_sigma2
        ) * torch.square(m_inv)
        r_var = (
            ray_radius
            * ray_radius
            * (
                (1.0 / 4) * d_mu2
                + (5.0 / 12) * d_sigma2
                - (4.0 / 15) * d_sigma4 * m_inv
            )
        )

        dir_sq = torch.square(sample_dir)
        diag_variance = t_var[:, :, None] * dir_sq + r_var[:, :, None] * (1.0 - dir_sq)

        sample_pos: Tensor = sample_orig + sample_dir * t_mu[:, :, None]
        sampling: Sampling = Sampling(
            sample_pos,
            sample_dir,
            diag_variance,
        )
        return sampling
