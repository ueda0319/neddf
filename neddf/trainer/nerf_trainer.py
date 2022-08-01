from pathlib import Path
from typing import Dict, Final, List

import cv2
import hydra
import numpy as np
import torch
from neddf.camera import BaseCameraCalib, Camera, PinholeCalib
from neddf.dataset import NeRFSyntheticDataset
from neddf.logger import NeRFTBLogger
from neddf.loss import BaseLoss
from neddf.network import BaseNeuralField
from neddf.render import NeRFRender, RenderTarget
from numpy import ndarray
from omegaconf import DictConfig
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


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
        network_fine: BaseNeuralField = hydra.utils.instantiate(
            self.config.network,
            is_coarse=False,
        ).to(self.device)
        network_coarse: BaseNeuralField = hydra.utils.instantiate(
            self.config.network,
            is_coarse=True,
        ).to(self.device)
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

        self.loss_functions: List[BaseLoss] = [
            hydra.utils.instantiate(loss_function).to(self.device)
            for loss_function in self.config.loss.functions
        ]

        self.optimizer: Adam = Adam(
            self.neural_render.get_parameters_list(),
            lr=self.config.trainer.optimizer_lr,
            weight_decay=self.config.trainer.optimizer_weight_decay,
        )
        self.scheduler: ExponentialLR = ExponentialLR(
            self.optimizer, gamma=self.config.trainer.scheduler_lr
        )
        self.logger = NeRFTBLogger()

    def load_pretrained_model(self, model_path: Path) -> None:
        self.neural_render.load_state_dict(torch.load(str(model_path)))  # type: ignore

    def run_train(self) -> None:
        # make directory in hydra's loggin directory for save models
        Path("models").mkdir(parents=True)
        render_dir: Path = Path("render")

        frame_length: Final[int] = len(self.dataset)
        for epoch in range(0, self.config.trainer.epoch_max + 1):
            print("epoch: ", epoch)
            self.neural_render.set_iter(epoch)
            camera_ids = np.random.permutation(frame_length)
            for camera_id in tqdm(camera_ids):
                self.run_train_step(camera_id)
            self.scheduler.step()
            # TODO parameterize epoch steps to logging in config (might be written in logger)
            if epoch % 2 == 0:
                output_field_dir: Path = render_dir / "fields"
                # make output directory if not exist
                output_field_dir.mkdir(parents=True, exist_ok=True)
                self.save_field_slice(output_field_dir, epoch)
            if epoch % 10 == 0:
                print("test rendering...")
                output_dir: Path = render_dir / "{:04}".format(epoch)
                output_dir.mkdir(parents=True)
                self.render_test(output_dir, camera_ids[0], downsampling=3)
            if epoch % 100 == 0:
                torch.save(
                    self.neural_render.state_dict(),
                    "models/model_{:0=5}.pth".format(epoch),
                )

    def run_train_step(self, camera_id: int) -> float:
        self.logger.write_batchstart()

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

        loss_types: List[str] = [type(func).__name__ for func in self.loss_functions]
        targets: Dict[str, Tensor] = {}
        if "ColorLoss" in loss_types:
            rgb_gt_np: ndarray = (1.0 / 256) * np.stack(
                [rgb[v, u, :] for u, v in zip(us_int, vs_int)]
            ).astype(np.float32)
            targets["color"] = (
                torch.from_numpy(rgb_gt_np).to(torch.float32).to(self.device)
            )

        if "MaskBCELoss" in loss_types:
            mask_gt_np: ndarray = (1.0 / 256) * np.stack(
                [mask[v, u] for u, v in zip(us_int, vs_int)]
            ).astype(np.float32)
            targets["mask"] = (
                torch.from_numpy(mask_gt_np).to(torch.float32).to(self.device)
            )

        if "AuxGradLoss" in loss_types:
            targets["aux_grad_penalty"] = torch.zeros_like(
                render_result["aux_grad_penalty"]
            )

        if "RangeLoss" in loss_types:
            targets["range_penalty"] = torch.zeros_like(render_result["range_penalty"])

        loss_dict: Dict[str, Tensor] = {}
        for loss_function in self.loss_functions:
            loss_dict.update(loss_function(render_result, targets))
        loss = torch.sum(torch.stack(list(loss_dict.values())))

        loss.backward()  # type: ignore
        loss_float = float(loss.item())
        self.logger.write(loss_float, 1.0, loss_dict)

        del loss
        del loss_dict

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.logger.write_batchend()
        self.logger.next()

        return loss_float

    def render_test(
        self, output_dir: Path, camera_id: int, downsampling: int = 1
    ) -> None:
        rgb = self.dataset[camera_id]["rgb_images"]
        camera = self.cameras[camera_id]
        camera.update_transform()
        h: Final[int] = rgb.shape[0]
        w: Final[int] = rgb.shape[1]
        target_types: List[RenderTarget] = ["color", "depth"]

        # render images with target_types "color" and "depth"
        images = self.neural_render.render_image(
            w, h, camera, target_types, downsampling, self.config.trainer.chunk
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
        frame_length: Final[int] = len(self.dataset)
        for camera_id in range(frame_length):
            print("rendering from camera {}".format(camera_id))
            self.render_test(output_dir, camera_id, 1)

    def save_field_slice(self, output_field_dir: Path, epoch: int = 0) -> None:
        images: Dict[str, ndarray] = self.neural_render.render_field_slice(self.device)
        for key in images:
            write_path: Path = output_field_dir / "field_{}_{:04}.png".format(key, epoch)
            cv2.imwrite(str(write_path), images[key])
