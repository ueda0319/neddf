from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Final, List, Literal

import cv2
import hydra
import numpy as np
import torch
from neddf.camera import BaseCameraCalib, Camera, PinholeCalib
from neddf.dataset import BaseDataset
from neddf.logger import BaseLogger
from neddf.loss import BaseLoss
from neddf.render import BaseNeuralRender, RenderTarget
from numpy import ndarray
from omegaconf import DictConfig
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

LossFunctionType = Literal[
    "ColorLoss", "MaskBCELoss", "MaskMSELoss", "FieldsConstraintLoss"
]


class BaseTrainer(ABC):
    """Abstract base class for Trainer."""

    config: DictConfig
    device: torch.device
    dataset: BaseDataset
    neural_render: BaseNeuralRender
    camera_calib: BaseCameraCalib
    cameras: List[Camera]
    loss_functions: List[BaseLoss]
    optimizer: Adam
    scheduler: ExponentialLR
    logger: BaseLogger
    # primitive
    batch_size: int
    chunk: int
    epoch_max: int
    epoch_save_fields: int
    epoch_test_rendering: int
    epoch_save_model: int
    scheduler_lr: float
    optimizer_lr: float
    optimizer_weight_decay: float

    def __init__(
        self,
        global_config: DictConfig,
        device: str = "cuda:0",
        batch_size: int = 1024,
        chunk: int = 1024,
        epoch_max: int = 2000,
        epoch_save_fields: int = 2,
        epoch_test_rendering: int = 10,
        epoch_save_model: int = 100,
        scheduler_lr: float = 0.99815,
        optimizer_lr: float = 0.0005,
        optimizer_weight_decay: float = 0.0,
    ) -> None:
        """Initializer

        This method initialize BaseTrainer module's common attributes.

        Args:
            global_config (DictConfig): Configuration data from hydra
            device (str): device information(cpu, cuda:0, etc.)
            batch_size (int): Batch size for training
            chunk (int): Chunk size for rendering
            epoch_max (int): Count of epoch for training
            epoch_save_fields (int): Frequency of save fields
            epoch_test_rendering (int): Frequency of test rendering
            epoch_save_model (int): Frequency of save model parameters
            scheduler_lr (float): Learning rate for scheduler
            optimizer_lr (float): Learning rate for optimizer
            optimizer_weight_decay (float): weight decay for optimizer
        """
        super().__init__()
        # Keep Hydra config data
        self.config = global_config
        # Configure torch device (cpu or cuda)
        self.device = torch.device(device)
        # Setup primitive parameters
        self.batch_size = batch_size
        self.chunk = chunk
        self.epoch_max = epoch_max
        self.epoch_save_fields = epoch_save_fields
        self.epoch_test_rendering = epoch_test_rendering
        self.epoch_save_model = epoch_save_model
        self.scheduler_lr = scheduler_lr
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        # Setup Dataset
        self.dataset = hydra.utils.instantiate(self.config.dataset)
        # Setup Cameras
        frame_length = len(self.dataset)
        self.camera_calib = PinholeCalib(self.dataset[0]["camera_calib_params"]).to(
            self.device
        )
        self.cameras = [
            Camera(self.camera_calib, self.dataset[camera_id]["camera_params"]).to(
                self.device
            )
            for camera_id in range(frame_length)
        ]
        # Setup Loss functions
        self.loss_functions = [
            hydra.utils.instantiate(loss_function).to(self.device)
            for loss_function in self.config.loss.functions
        ]

    def load_pretrained_model(self, model_path: Path) -> None:
        """Load pretrained model

        Args:
            model_path (Path): Path to model parameter file
        """
        self.neural_render.load_state_dict(torch.load(str(model_path)))  # type: ignore

    def render_test(
        self, output_dir: Path, camera_id: int, downsampling: int = 1
    ) -> None:
        """Render test

        This method render image

        Args:
            output_dir (Path): Path to save directory
            camera_id (int): Selected camera id
            downsampling (int): Step of downsampling(0 for full render)
        """
        rgb_gt = self.dataset[camera_id]["rgb_images"].astype(np.uint8)
        camera = self.cameras[camera_id]
        camera.update_transform()
        h: Final[int] = rgb_gt.shape[0]
        w: Final[int] = rgb_gt.shape[1]
        target_types: List[RenderTarget] = ["color", "depth"]

        # render images with target_types "color" and "depth"
        images = self.neural_render.render_image(
            w, h, camera, target_types, downsampling, self.chunk
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
        """Render from all camera

        This method render image from all cameras

        Args:
            output_dir (Path): Path to save directory
        """
        frame_length: Final[int] = len(self.dataset)
        for camera_id in range(frame_length):
            print("rendering from camera {}".format(camera_id))
            self.render_test(output_dir, camera_id, 1)

    def render_field_slices(self, output_field_dir: Path, epoch: int = 0) -> None:
        """Render field slices

        This method visualize field by slice

        Args:
            output_dir (Path): Path to save directory
            epoch (int): Current epoch for filename
        """
        images: Dict[str, ndarray] = self.neural_render.render_field_slice()
        for key in images:
            write_path: Path = output_field_dir / "field_{}_{:04}.png".format(
                key, epoch
            )
            cv2.imwrite(str(write_path), images[key])

    def construct_ground_truth(
        self,
        camera_id: int,
        us_int: Tensor,
        vs_int: Tensor,
        loss_types: List[LossFunctionType],
    ) -> Dict[str, Tensor]:
        """Construct ground truth

        This method construct target values for use loss

        Args:
            camera_id (int): Selected camera index
            us_int (Tensor[batch_size, int]): Vertical pixel position of rays
            vs_int (Tensor[batch_size, int]): Horizontal pixel position of rays
            loss_types (List[LossFunctionType]): Selected loss functions
        """
        targets: Dict[str, Tensor] = {}
        if "ColorLoss" in loss_types:
            rgb = self.dataset[camera_id]["rgb_images"]
            rgb_gt_np: ndarray = (1.0 / 256) * np.stack(
                [rgb[v, u, :] for u, v in zip(us_int, vs_int)]
            ).astype(np.float32)
            targets["color"] = (
                torch.from_numpy(rgb_gt_np).to(torch.float32).to(self.device)
            )

        if "MaskBCELoss" in loss_types or "MaskMSELoss" in loss_types:
            mask = self.dataset[camera_id]["mask_images"]
            mask_gt_np: ndarray = (1.0 / 256) * np.stack(
                [mask[v, u] for u, v in zip(us_int, vs_int)]
            ).astype(np.float32)
            targets["mask"] = (
                torch.from_numpy(mask_gt_np).to(torch.float32).to(self.device)
            )

        if "FieldsConstraintLoss" in loss_types:
            targets["fields_penalty"] = torch.zeros(us_int.shape, dtype=torch.float32)

        return targets

    @abstractmethod
    def run_train(self) -> None:
        """Run train

        This method execute training
        """
        raise NotImplementedError()

    @abstractmethod
    def run_train_step(self, camera_id: int) -> float:
        """Run train step

        This method execute one step of training

        Args:
            camera_id (int): Selected camera index
        """
        raise NotImplementedError()
