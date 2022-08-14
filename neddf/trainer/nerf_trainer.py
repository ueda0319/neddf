import math
from pathlib import Path
from typing import Dict, Final, List

import hydra
import numpy as np
import torch
from neddf.logger import NeRFTBLogger
from neddf.trainer.base_trainer import BaseTrainer, LossFunctionType
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


class NeRFTrainer(BaseTrainer):
    """NeRFTrainer module.

    Attributes:
        config (DictConfig): experiment setting get from hydra
    """

    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        """Initializer

        This method initialize NeRFTrainer class
        """
        super().__init__(**kwargs)
        self.neural_render = hydra.utils.instantiate(
            self.config.render,
            network_config=self.config.network,
            _recursive_=False,
        ).to(self.device)
        # Setup Optimizer
        self.optimizer = Adam(
            self.neural_render.get_parameters_list(),
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )
        self.scheduler = ExponentialLR(self.optimizer, gamma=self.scheduler_lr)
        # Use tensorboard Logger
        self.logger = NeRFTBLogger()

    def run_train(self) -> None:
        """Run train

        This method execute training
        """
        # make directory in hydra's loggin directory for save models
        Path("models").mkdir(parents=True)
        render_dir: Path = Path("render")

        frame_length: Final[int] = len(self.dataset)
        self.neural_render.set_iter(0)
        for epoch in range(0, self.epoch_max + 1):
            print("epoch: ", epoch)
            camera_ids = np.random.permutation(frame_length)
            for camera_id in tqdm(camera_ids):
                self.run_train_step(camera_id)
                self.neural_render.next_iter()
            self.scheduler.step()
            if epoch % self.epoch_save_fields == 0:
                output_field_dir: Path = render_dir / "fields"
                # make output directory if not exist
                output_field_dir.mkdir(parents=True, exist_ok=True)
                self.render_field_slices(output_field_dir, epoch)
            if epoch % self.epoch_test_rendering == 0:
                print("test rendering...")
                output_dir: Path = render_dir / "{:04}".format(epoch)
                output_dir.mkdir(parents=True)
                self.render_test(output_dir, camera_ids[0], downsampling=3)
            if epoch % self.epoch_save_model == 0:
                torch.save(
                    self.neural_render.state_dict(),
                    "models/model_{:0=5}.pth".format(epoch),
                )

    def run_train_step(self, camera_id: int) -> float:
        """Run train step

        This method execute one step of training

        Args:
            camera_id (int): Selected camera index
        """
        self.logger.write_batchstart()

        self.optimizer.zero_grad()
        batch_size: Final[int] = self.batch_size
        rgb = self.dataset[camera_id]["rgb_images"]
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

        loss_types: List[LossFunctionType] = [
            type(func).__name__ for func in self.loss_functions  # type: ignore
        ]
        targets: Dict[str, Tensor] = self.construct_ground_truth(
            camera_id, us_int, vs_int, loss_types
        )

        loss_dict: Dict[str, Tensor] = {}
        for loss_function in self.loss_functions:
            loss_dict.update(loss_function(render_result, targets))
        loss = torch.sum(torch.stack(list(loss_dict.values())))

        loss.backward()  # type: ignore
        loss_float = float(loss.item())

        mse: float = float(
            torch.mean(torch.square(render_result["color"] - targets["color"])).item()
        )
        psnr = 10 * math.log10(1.0 / mse)
        self.logger.write(loss_float, psnr, loss_dict)

        del loss
        del loss_dict

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.logger.write_batchend()
        self.logger.next()

        return loss_float
