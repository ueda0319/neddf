from pathlib import Path
from typing import Dict, Final, List

import cv2
import hydra
import numpy as np
import torch
from numpy import ndarray
from omegaconf import DictConfig
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from melon.camera import BaseCameraCalib, Camera, PinholeCalib
from melon.dataset import NeRFSyntheticDataset
from melon.network import NeRF
from melon.render import NeRFRender, RenderTarget


class NeRFTrainer:
    """NeRFTrainer module.

    Attributes:
        config (DictConfig): experiment setting get from hydra
    """

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        super().__init__()
        self.config: DictConfig = config
        self.device: torch.device = torch.device(self.config.trainer.device)
        # TODO change types to union
        self.dataset: NeRFSyntheticDataset = hydra.utils.instantiate(
            self.config.dataset
        )
        network_coarse: NeRF = hydra.utils.instantiate(self.config.network).to(
            self.device
        )
        network_fine: NeRF = hydra.utils.instantiate(self.config.network).to(
            self.device
        )
        self.neural_render: NeRFRender = hydra.utils.instantiate(
            self.config.render,
            network_coarse=network_coarse,
            network_fine=network_fine,
        ).to(self.device)

        frame_length: Final[int] = len(self.dataset)
        self.camera_calib: BaseCameraCalib = PinholeCalib(
            self.dataset[0]["camera_calib_params"]
        ).to(self.device)

        self.cameras: List[Camera] = [
            Camera(self.camera_calib, self.dataset[camera_id]["camera_params"]).to(
                self.device
            )
            for camera_id in range(frame_length)
        ]

        self.optimizer: Adam = Adam(
            self.neural_render.get_parameters_list(),
            lr=self.config.trainer.optimizer_lr,
            weight_decay=self.config.trainer.optimizer_weight_decay,
        )
        self.scheduler: ExponentialLR = ExponentialLR(
            self.optimizer, gamma=self.config.trainer.scheduler_lr
        )

    def load_pretrained_model(self, model_path: Path) -> None:
        self.neural_render.load_state_dict(torch.load(str(model_path)))  # type: ignore

    def run_train(self) -> None:
        # make directory in hydra's loggin directory for save models
        Path("models").mkdir(parents=True)

        frame_length: Final[int] = len(self.dataset)
        for epoch in range(1, self.config.trainer.epoch_max + 1):
            print("epoch: ", epoch)
            camera_ids = np.random.permutation(frame_length)
            for camera_id in tqdm(camera_ids):
                self.run_train_step(camera_id)
            self.scheduler.step()
            # TODO parameterize epoch steps to logging in config (might be written in logger)
            if epoch % 10 == 0:
                print("test rendering...")
                output_dir: Path = Path("render/{:04}".format(epoch))
                output_dir.mkdir(parents=True)
                self.render_test(output_dir, camera_ids[0], downsampling=2)
            if epoch % 100 == 0:
                torch.save(
                    self.neural_render.state_dict(),
                    "models/model_{:0=5}.pth".format(epoch),
                )

    def run_train_step(self, camera_id: int) -> float:
        self.optimizer.zero_grad()
        batch_size: Final[int] = self.config.trainer.batch_size
        rgb = self.dataset[camera_id]["rgb_images"]
        mask = self.dataset[camera_id]["mask_images"]
        camera = self.cameras[camera_id]

        camera.update_transform()
        h: Final[int] = rgb.shape[0]
        w: Final[int] = rgb.shape[1]

        us_int: Tensor = (
            (torch.rand(batch_size) * (w - 1)).to(torch.int16).to(self.device)
        )
        vs_int: Tensor = (
            (torch.rand(batch_size) * (h - 1)).to(torch.int16).to(self.device)
        )
        uv: Tensor = torch.stack([us_int, vs_int], 1)

        render_result: Dict[str, Tensor] = self.neural_render.render_rays(uv, camera)
        rgb_result: Tensor = render_result["color"]
        rgb_coarse_result: Tensor = render_result["color_coarse"]
        mask_result: Tensor = torch.clamp(
            1.0 - render_result["transmittance"], 1e-6, 1.0 - 1e-6
        )
        mask_coarse_result: Tensor = torch.clamp(
            1.0 - render_result["transmittance"], 1e-6, 1.0 - 1e-6
        )

        rgb_gt_np: ndarray = (1.0 / 256) * np.stack(
            [rgb[v, u, :] for u, v in zip(us_int, vs_int)]
        ).astype(np.float32)
        mask_gt_np: ndarray = (1.0 / 256) * np.stack(
            [mask[v, u] for u, v in zip(us_int, vs_int)]
        ).astype(np.float32)
        rgb_gt: Tensor = torch.from_numpy(rgb_gt_np).to(torch.float32).to(self.device)
        mask_gt: Tensor = torch.from_numpy(mask_gt_np).to(torch.float32).to(self.device)

        # calculate color loss
        loss_color: Tensor = torch.mean(torch.square(rgb_result - rgb_gt))
        loss_color_coarse: Tensor = torch.mean(torch.square(rgb_coarse_result - rgb_gt))
        # calculate mask loss
        loss_mask: Tensor = -torch.mean(
            mask_gt * torch.log(mask_result)
            + (1.0 - mask_gt) * torch.log(1.0 - mask_result)
        )
        loss_mask_coarse: Tensor = -torch.mean(
            mask_gt * torch.log(mask_coarse_result)
            + (1.0 - mask_gt) * torch.log(1.0 - mask_coarse_result)
        )

        # parameters of mixing ratio
        lambda_color: Final[float] = self.config.trainer.lambda_color
        lambda_mask: Final[float] = self.config.trainer.lambda_mask
        lambda_coarse: Final[float] = self.config.trainer.lambda_coarse
        # mix loss
        loss = lambda_color * (
            loss_color + lambda_coarse * loss_color_coarse
        ) + lambda_mask * (loss_mask + lambda_coarse * loss_mask_coarse)

        loss.backward()  # type: ignore
        self.optimizer.step()

        return float(loss.item())

    def render_test(
        self, output_dir: Path, camera_id: int, downsampling: int = 1
    ) -> None:
        with torch.no_grad():  # type: ignore
            rgb = self.dataset[camera_id]["rgb_images"]
            camera = self.cameras[camera_id]
            camera.update_transform()
            h: Final[int] = rgb.shape[0]
            w: Final[int] = rgb.shape[1]
            target_types: List[RenderTarget] = ["color", "depth"]

            # render images with target_types "color" and "depth"
            images = self.neural_render.render_image(
                w, h, camera, target_types, downsampling
            )
            rgb_np = (
                torch.clamp(images["color"] * 255, 0, 255)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            depth_np = (
                torch.clamp((images["depth"] - 2.0) / 4.0 * 50000 / 256, 0, 255)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            rgb_gt = self.dataset[camera_id]["rgb_images"]

            # output_pathes
            rgb_path: Path = output_dir / "{:03}_rgb.png".format(camera_id)
            depth_path: Path = output_dir / "{:03}_depth.png".format(camera_id)
            rgb_gt_path: Path = output_dir / "{:03}_rgb_gt.png".format(camera_id)
            # write images
            cv2.imwrite(str(rgb_path), rgb_np)
            cv2.imwrite(str(rgb_gt_path), rgb_gt)
            cv2.imwrite(str(depth_path), depth_np)

            # write psnr and ssim on fullsize-rendering
            if downsampling == 1:
                psnr = peak_signal_noise_ratio(rgb_np, rgb_gt)
                ssim = structural_similarity(rgb_np, rgb_gt, channel_axis=2)
                print("psnr: {}, ssim: {}".format(psnr, ssim))

    def render_all(self, output_dir: Path) -> None:
        with torch.no_grad():  # type: ignore
            frame_length: Final[int] = len(self.dataset)
            for camera_id in range(frame_length):
                print("rendering from camera {}".format(camera_id))
                self.render_test(output_dir, camera_id, 1)
