import math
from typing import Any, Dict, Final, Iterable, List, Literal

import cv2
import hydra
import numpy as np
import torch
from neddf.camera import Camera
from neddf.network import BaseNeuralField
from neddf.ray import Ray, Sampling
from neddf.render.base_neural_render import BaseNeuralRender, RenderTarget
from numpy import ndarray
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm

SamplingType = Literal["point", "cone"]


class NeRFRender(BaseNeuralRender):
    def __init__(
        self,
        network_config: DictConfig,
        sample_coarse: int = 128,
        sample_fine: int = 128,
        dist_near: float = 2.0,
        dist_far: float = 6.0,
        max_dist: float = 6.0,
        use_coarse_network: bool = True,
        sampling_type: SamplingType = "point",
    ) -> None:
        super().__init__()
        self.use_coarse_network: bool = use_coarse_network
        self.network_fine: BaseNeuralField = hydra.utils.instantiate(
            network_config,
        )
        if use_coarse_network:
            self.network_coarse: BaseNeuralField = hydra.utils.instantiate(
                network_config,
            )
        else:
            self.network_coarse = self.network_fine
        self.sample_coarse: Final[int] = sample_coarse
        self.sample_fine: Final[int] = sample_fine
        self.dist_near: Final[float] = dist_near
        self.dist_far: Final[float] = dist_far
        self.max_dist: Final[float] = max_dist
        self.sampling_type: SamplingType = sampling_type

    def get_parameters_list(self) -> List[Any]:
        if self.use_coarse_network:
            return list(self.network_coarse.parameters()) + list(
                self.network_fine.parameters()
            )
        else:
            return list(self.network_coarse.parameters())

    def render_rays(
        self,
        uv: Tensor,
        camera: Camera,
    ) -> Dict[str, Tensor]:
        batch_size = uv.shape[0]
        device = uv.device

        rays: Ray = camera.create_rays(uv)

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
        if self.sampling_type == "point":
            samples_coarse: Sampling = rays.get_sampling_points(dists_coarse)
        elif self.sampling_type == "cone":
            # use 1111 for view angle = 0.6911112070083618(rad)
            ray_radius: float = 1.0 / 1111 / math.sqrt(12)
            samples_coarse = rays.get_sampling_cones(dists_coarse, ray_radius)
        values_coarse: Dict[str, Tensor] = self.network_coarse(samples_coarse)
        integrate_coarse: Dict[str, Tensor] = self.integrate_volume_render(
            dists=dists_coarse,
            densities=values_coarse["density"],
            colors=values_coarse["color"],
        )
        for key in values_coarse:
            if "penalty" in key:
                integrate_coarse[key] = torch.sum(
                    delta_coarse.detach()
                    * values_coarse[key].reshape(batch_size, -1)[:, :-1],
                    dim=1,
                )

        with torch.no_grad():  # type: ignore
            dists_fine: Tensor = self.sample_pdf(
                dists_coarse,
                integrate_coarse["weight"],
                self.sample_fine + 1,
            )
        delta_fine: Tensor = dists_fine[:, 1:] - dists_fine[:, :-1]
        if self.sampling_type == "point":
            samples_fine: Sampling = rays.get_sampling_points(dists_fine)
        elif self.sampling_type == "cone":
            samples_fine = rays.get_sampling_cones(dists_fine, ray_radius)
        values_fine: Dict[str, Tensor] = self.network_fine(samples_fine)
        integrate: Dict[str, Tensor] = self.integrate_volume_render(
            dists=dists_fine,
            densities=values_fine["density"],
            colors=values_fine["color"],
        )
        for key in values_fine:
            if "penalty" in key:
                integrate[key] = torch.sum(
                    delta_fine.detach()
                    * values_fine[key].reshape(batch_size, -1)[:, :-1],
                    dim=1,
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
                key: torch.cat(integrates[key], 0).reshape(h, w, -1)
                for key in target_types
            }
            self.network_coarse.train(True)
            self.network_fine.train(True)
        return images

    def set_iter(self, iter: int) -> None:
        """Set iteration

        This methods set iteration number for configure warm ups

        Args:
            iter (int): current iteration. Set -1 for evaluation.
        """
        super().set_iter(iter)
        self.network_coarse.set_iter(iter)
        self.network_fine.set_iter(iter)

    def render_field_slice(
        self,
        device: torch.device,
        slice_t: float = 0.0,
        render_size: float = 1.1,
        render_resolution: int = 128,
    ) -> Dict[str, ndarray]:
        with torch.set_grad_enabled(False):
            xs = (
                torch.linspace(-render_size, render_size, render_resolution)
                .to(device)
                .reshape(1, render_resolution)
                .expand(render_resolution, render_resolution)
            )
            ys = -(
                torch.linspace(-render_size, render_size, render_resolution)
                .to(device)
                .reshape(render_resolution, 1)
                .expand(render_resolution, render_resolution)
            )
            zs = torch.zeros(render_resolution, render_resolution).to(device) + slice_t

            sample_pos = torch.cat(
                [xs[:, :, None], ys[:, :, None], zs[:, :, None]], dim=2
            )
            sample_dir = torch.zeros(render_resolution, render_resolution, 3).to(device)
            sample_dir[:, :, 2] = 1.0
            diag_variance = torch.zeros_like(sample_pos)
            sampling: Sampling = Sampling(
                sample_pos,
                sample_dir,
                diag_variance,
            )
            self.network_fine.train(False)
            values: Dict[str, Tensor] = self.network_fine(sampling)
            self.network_fine.train(True)
            fields: Dict[str, ndarray] = {}
            scales: Dict[str, float] = {
                "distance": 256.0,
                "density": 12.8,
                "color": 256.0,
                "aux_grad": 256.0,
            }
            for value_type in values:
                if value_type not in scales:
                    continue
                field: ndarray = (
                    scales[value_type]
                    * values[value_type]
                    .reshape(render_resolution, render_resolution, -1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                if field.shape[2] == 1:
                    field_cv: ndarray = cv2.applyColorMap(
                        field.clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET
                    )
                else:
                    field_cv = field.clip(0, 255).astype(np.uint8)
                fields[value_type] = field_cv
            return fields
