from typing import Any, Dict, Final, Iterable, List, Literal

import torch
from torch import Tensor
from torch.nn.functional import relu
from tqdm import tqdm

from neddf.camera import Camera
from neddf.network import BaseNeuralField
from neddf.render.base_neural_render import BaseNeuralRender

RenderTarget = Literal["color", "depth", "transmittance"]


class NeRFRender(BaseNeuralRender):
    def __init__(
        self,
        network_coarse: BaseNeuralField,
        network_fine: BaseNeuralField,
        sample_coarse: int = 128,
        sample_fine: int = 128,
        dist_near: float = 2.0,
        dist_far: float = 6.0,
        max_dist: float = 6.0,
    ) -> None:
        super().__init__()
        self.network_coarse: BaseNeuralField = network_coarse
        self.network_fine: BaseNeuralField = network_fine
        self.sample_coarse: Final[int] = sample_coarse
        self.sample_fine: Final[int] = sample_fine
        self.dist_near: Final[float] = dist_near
        self.dist_far: Final[float] = dist_far
        self.max_dist: Final[float] = max_dist

    def get_parameters_list(self) -> List[Any]:
        return list(self.network_coarse.parameters()) + list(
            self.network_fine.parameters()
        )

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

    def render_rays(
        self,
        uv: Tensor,
        camera: Camera,
    ) -> Dict[str, Tensor]:
        batch_size = uv.shape[0]
        device = uv.device

        rays = camera.create_rays(uv)

        # sample distances linearly for coarse sampling
        dists_coarse: Tensor = (
            torch.linspace(self.dist_near, self.dist_far, self.sample_coarse + 1)
            .to(device)
            .reshape(1, self.sample_coarse + 1)
            .expand(batch_size, self.sample_coarse + 1)
        ) + (
            torch.rand(batch_size, self.sample_coarse + 1).to(device)
            * ((self.dist_far - self.dist_near) / (self.sample_coarse))
        )
        delta_coarse: Tensor = dists_coarse[:, 1:] - dists_coarse[:, :-1]
        samples_coarse: Dict[str, Tensor] = rays.get_sampling_points(dists_coarse)
        values_coarse: Dict[str, Tensor] = self.network_coarse(
            samples_coarse["sample_pos"],
            samples_coarse["sample_dir"],
        )
        integrate_coarse: Dict[str, Tensor] = self.integrate_volume_render(
            dists=dists_coarse,
            densities=values_coarse["density"],
            colors=values_coarse["color"],
        )
        for key in values_coarse:
            if "penalty" in key:
                integrate_coarse[key] = torch.sum(
                    delta_coarse.detach() * values_coarse[key].reshape(batch_size, -1)[:, :-1],
                    dim=1
                )

        with torch.no_grad():  # type: ignore
            dists_fine: Tensor = self.sample_pdf(
                dists_coarse,
                integrate_coarse["weight"],
                self.sample_fine + 1,
            )
        delta_fine: Tensor = dists_fine[:, 1:] - dists_fine[:, :-1]
        samples_fine: Dict[str, Tensor] = rays.get_sampling_points(dists_fine)
        values_fine: Dict[str, Tensor] = self.network_fine(
            samples_fine["sample_pos"],
            samples_fine["sample_dir"],
        )
        integrate: Dict[str, Tensor] = self.integrate_volume_render(
            dists=dists_fine,
            densities=values_fine["density"],
            colors=values_fine["color"],
        )
        for key in values_fine:
            if "penalty" in key:
                integrate[key] = torch.sum(
                    delta_fine.detach() * values_fine[key].reshape(batch_size, -1)[:, :-1],
                    dim=1
                )
        for key in integrate_coarse:
            key_coarse = "{}_coarse".format(key)
            integrate[key_coarse] = integrate_coarse[key]
        return integrate

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

        with torch.set_grad_enabled(False):
            torch.cuda.empty_cache()
            w = width // downsampling
            h = height // downsampling
            us = (
                torch.arange(w).reshape(1, w).expand(h, w).reshape(-1).to(camera.device)
                * downsampling
            )
            vs = (
                torch.arange(h).reshape(h, 1).expand(h, w).reshape(-1).to(camera.device)
                * downsampling
            )
            uv: Tensor = torch.stack([us, vs], 1)

            pixel_length = us.shape[0]
            integrates: Dict[str, List[Tensor]] = {key: [] for key in target_types}

            self.network_coarse.eval()
            self.network_fine.eval()
            for below in tqdm(range(0, pixel_length, chunk)):
                above = min(pixel_length, below + chunk)
                integrate = self.render_rays(uv[below:above, :], camera)
                for key in target_types:
                    integrates[key].append(integrate[key].detach())
                
            images: Dict[str, Tensor] = {
                key: torch.cat(integrates[key], 0).reshape(h, w, -1) for key in target_types
            }
            self.network_coarse.train(True)
            self.network_fine.train(True)
        return images
