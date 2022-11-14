import math
from pathlib import Path
from typing import Dict, Final, List

import hydra
import numpy as np
import torch
from neddf.camera import Camera
from neddf.logger import NeRFTBLogger
from neddf.trainer.base_trainer import BaseTrainer, LossFunctionType
from numpy import ndarray
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

class NeDDFTracker(BaseTrainer):
    def __init__(self) -> None:
        pass
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
        # Use tensorboard Logger
        self.logger = NeRFTBLogger()

    def run_track_all(self) -> None:
        pass
        

    def run_track_photometric_step(self, camera_id: int, target_camera: Camera) -> float:
        self.logger.write_batchstart()

        self.optimizer.zero_grad()
        batch_size: Final[int] = self.batch_size
        rgb = self.dataset[camera_id]["rgb_images"]
        camera = self.cameras[camera_id]

        target_camera.update_transform()
        h: Final[int] = rgb.shape[0]
        w: Final[int] = rgb.shape[1]

        us_int: Tensor = (
            (torch.rand(batch_size) * (w - 1)).to(torch.int16).to(self.device)
        )
        vs_int: Tensor = (
            (torch.rand(batch_size) * (h - 1)).to(torch.int16).to(self.device)
        )
        uv: Tensor = torch.stack([us_int, vs_int], 1)

        rgb_gt_np: ndarray = (1.0 / 256) * np.stack(
            [rgb[v, u, :] for u, v in zip(us_int, vs_int)]
        ).astype(np.float32)
        rgb_gt: Tensor = torch.from_numpy(rgb_gt_np).to(torch.float32).to(self.device)

        render_result: Dict[str, Tensor] = self.neural_render.render_rays(uv, target_camera)

        loss_types: List[LossFunctionType] = [
            type(func).__name__ for func in self.loss_functions  # type: ignore
        ]
        targets: Dict[str, Tensor] = self.construct_ground_truth(
            camera_id, us_int, vs_int, loss_types
        )
        
        loss = torch.mean(torch.square(render_result["color"] - rgb_gt))

        loss.backward()  # type: ignore
        loss_float = float(loss.item())

        mse: float = float(
            torch.mean(torch.square(render_result["color"] - targets["color"])).item()
        )
        psnr = 10 * math.log10(1.0 / mse)
        self.logger.write(loss_float, psnr, {"color": loss})

        del loss

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.logger.write_batchend()
        self.logger.next()

        return loss_float