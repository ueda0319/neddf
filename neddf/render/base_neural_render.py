from abc import ABC, abstractmethod
from typing import Any, Dict, Final, Iterable, List, Literal

import torch
from neddf.camera import Camera
from numpy import ndarray
from torch import Tensor, nn
from torch.nn.functional import relu

RenderTarget = Literal["color", "depth", "transmittance"]


class BaseNeuralRender(ABC, nn.Module):
    # Hierarchical sampling
    def sample_pdf(
        self,
        dists: Tensor,
        weights: Tensor,
        samples_fine: int,
        cat_coarse: bool = True,
    ) -> Tensor:
        """PDF sampling

        This method generate fine sampling points by pdf sampling

        Args:
            dists (Tensor[batch_size, samples_coarse, float32]):
                distance of each coarse sampling points from camera position
            weights (Tensor[batch_size, samples_coarse, float32]):
                weight of each coarse sampling points
            samples_fine (int): count of sampling points this method generate
            cat_coarse (bool):
                flag to concatenate coarse sampling point in output fine samplings

        Returns:
            Tensor: dists of fine sampling points
                note that shape of output takes [batch_size, samples_coarse + samples_fine] when
                cat_coarse is True, and [batch_size, samples_fine] other
        """
        if torch.any(torch.isnan(weights)) or torch.any(weights < 0.0):
            print("sample_pdf: Invalid weight detected.")
            weights[weights < 0.0] *= 0.0
            weights[torch.isnan(weights)] = 0.0

        # Get pdf
        weights = weights + 1e-2
        batch_size = dists.shape[0]
        device = dists.device
        if not cat_coarse:
            w1, _ = torch.max(
                torch.cat([weights[:, 2:, None], weights[:, 1:-1, None]], -1), dim=2
            )
            w2, _ = torch.max(
                torch.cat([weights[:, :-2, None], weights[:, 1:-1, None]], -1), dim=2
            )
            weights[:, 1:-1] = 0.5 * (w1 + w2)
        # Probability Distribution Function(PDF) from coarse sampling
        pdf = torch.nn.functional.normalize(weights, p=1.0, dim=-1)
        # Cumulative Distribution Function(CDF) from coarse sampling
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
        # Unif[0, 1]
        uniform_rands = torch.rand(batch_size, samples_fine).to(device).contiguous()

        # Invert CDF from Unif[0, 1]
        # Get the indexes of the interval in the CDF
        ids = torch.searchsorted(cdf, uniform_rands, right=True)
        # calculate left and right side of the interval
        below = torch.max(torch.zeros_like(ids), ids - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(ids), ids)
        # Get indexes of left and right side of each interval for gather cdf and distances
        # Note that subscript `_g` means `torch gather`
        ids_g = torch.stack([below, above], -1)
        matched_shape = [ids_g.shape[0], ids_g.shape[1], cdf.shape[-1]]
        # Gather cdf values of left and right side of each interval
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, ids_g)
        # Gather distance values of left and right side of each interval
        dists_g = torch.gather(dists.unsqueeze(1).expand(matched_shape), 2, ids_g)

        # Get the interval-size
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        # Interior
        t = (uniform_rands - cdf_g[..., 0]) / denom
        samples = dists_g[..., 0] + t * (dists_g[..., 1] - dists_g[..., 0])

        # Concatenate coarse sampling points when the flag is enabled.
        if cat_coarse:
            samples_cat, _ = torch.sort(torch.cat([samples, dists], -1), dim=-1)
        else:
            samples_cat, _ = torch.sort(samples, dim=-1)

        if torch.any(torch.isnan(samples_cat)):
            print("pdf sampling failed")
            samples_cat = (
                torch.linspace(
                    float(dists[0, 0]), float(dists[0, -1]), samples_cat.shape[1]
                )
                .to(device)
                .reshape(1, -1)
                .expand(batch_size, -1)
            )
        return samples_cat

    def integrate_volume_render(
        self,
        dists: Tensor,
        densities: Tensor,
        colors: Tensor,
    ) -> Dict[str, Tensor]:
        batch_size: Final[int] = dists.shape[0]
        sampling_step: Final[int] = dists.shape[1]

        device: Final[torch.device] = dists.device
        deltas = dists[:, 1:] - dists[:, :-1]

        o = 1 - torch.exp(-relu(densities[:, :-1]) * deltas)
        t = torch.cumprod(
            torch.cat([torch.ones((o.shape[0], 1)).to(device), 1.0 - o + 1e-7], 1), 1
        )
        w = o * t[:, :-1]
        assert not torch.any(torch.isnan(w))

        dh = w * dists[:, :-1]
        ih = w.reshape(batch_size, -1, 1).expand(batch_size, -1, 3) * colors[:, :-1, :]
        d = torch.sum(dh, dim=1)
        i = torch.sum(ih, dim=1)
        dv = torch.sum(
            w
            * torch.square(
                dists[:, :-1]
                - d.reshape(batch_size, 1).expand(batch_size, sampling_step - 1)
            ),
            dim=1,
        )

        # Black background
        d += t[:, -1] * self.max_dist

        # TODO: Make dataclass for volumerender result after append other neuralrender models
        result = {
            "weight": w,
            "depth": d,
            "depth_var": dv,
            "color": i,
            "transmittance": t[:, -1],
        }
        return result

    @abstractmethod
    def get_parameters_list(self) -> List[Any]:
        raise NotImplementedError()

    @abstractmethod
    def render_rays(
        self,
        uv: Tensor,
        camera: Camera,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def render_image(
        self,
        width: int,
        height: int,
        camera: Camera,
        target_types: Iterable[RenderTarget],
        downsampling: int = 1,
        chunk: int = 512,
    ) -> Dict[str, Tensor]:
        """render image

        Render image from selected camera.

        Args:
            width (int): width of original image
            height (int): height of original image
            camera (Camera): rendering images from this camera
            target_types (Iterable[RenderTarget]): type of rendering targets.
                select a subset of ["color", "depth", "transmittance"]
            downsampling (int): level of downsampling.
                For example, specifying 2 reduces the width and height of rendereing by half.
            chunk (int): number of rays to calculate at one step.
                Select according to GPU memory size.

        Returns:
            Dict[str, Tensor]: each type of images which selected in target_types.
        """
        raise NotImplementedError()

    @abstractmethod
    def render_field_slice(
        self,
        device: torch.device,
        slice_t: float = 0.0,
        render_size: float = 1.1,
        render_resolution: int = 128,
    ) -> Dict[str, ndarray]:
        raise NotImplementedError()

    def set_iter(self, iter: int) -> None:
        """Set iteration

        This methods set iteration number for configure warm ups

        Args:
            iter (int): current iteration. Set -1 for evaluation.
        """
        pass
