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
from scipy.spatial.transform import Rotation
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

    def run_train(self) -> None:
        pass
    def run_train_step(self, camera_id: int) -> float:
        pass

    def RGB2YUV(self, rgbs: Tensor) -> Tensor:
        mat = torch.tensor(
            [[0.2126,0.7152,0.0722],
            [-0.09991, -0.33609, 0.436],
            [0.001626, -0.55861, -0.05639]]
        ).t().to(torch.float32).to(self.device)
        mat = torch.tensor(
            [[0.299,0.587,0.114],
            [-0.169, -0.3316, 0.500],
            [0.500, -0.4186, -0.0813]]
        ).t().to(torch.float32).to(self.device)
        bgr2rgb = torch.tensor(
            [[0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0]]
        ).to(torch.float32).to(self.device)
        return torch.matmul(rgbs.reshape(-1,1,3), torch.matmul(bgr2rgb, mat))

    def generate_random_dir(self) -> ndarray:
        u = np.random.rand()
        v = np.random.rand()
        z = -2*u+1
        x=np.sqrt(1-z*z)*np.cos(2*np.pi*v)
        y=np.sqrt(1-z*z)*np.sin(2*np.pi*v)
        return np.array([x,y,z])
        
    def generate_initial_camera_poses(self) -> None:
        var_pos = 0.1
        var_rot = 0.1

        self.gt_camera_params: List[ndarray] = []
        self.initial_camera_params: List[ndarray] = []
        self.target_cameras: List[Camera] = []
        for camera in self.cameras:
            params_gt = camera.initial_params_np
            pos_gt = params_gt[3:]
            rot_gt = Rotation.from_rotvec(params_gt[:3])
            self.gt_camera_params.append(params_gt)

            rot_noise: Rotation = Rotation.from_rotvec(self.generate_random_dir() * np.random.randn(1) * var_rot)
            params =  np.zeros(6, np.float32)
            params[:3] = (rot_noise * rot_gt).as_rotvec()
            params[3:] = rot_noise.apply(pos_gt) + self.generate_random_dir() * np.random.randn(1) * var_pos
            self.initial_camera_params.append(params)
            target_camera: Camera =  Camera(self.camera_calib, params).to(self.device)
            self.target_cameras.append(target_camera)

    def evaluate_camera_pose(self, source_camera, target_camera) -> Dict[str, float]:
        source_camera.update_transform()
        target_camera.update_transform()
        source_rot = Rotation.from_matrix(source_camera.R.detach().cpu().numpy())
        target_rot = Rotation.from_matrix(target_camera.R.detach().cpu().numpy())
        source_trans = source_camera.T.detach().cpu().numpy()
        target_trans = target_camera.T.detach().cpu().numpy()

        rot_diff = source_rot.inv() * target_rot
        rot_err_rad: float = np.sqrt(np.sum(np.square(rot_diff.as_rotvec())))
        rot_err_deg: float = rot_err_rad * 180.0 / 3.141592

        trans_err: float = np.sqrt(np.sum(np.square(source_trans - target_trans)))

        errors: Dict[str, float] = {
            "rot_err": rot_err_deg,
            "trans_err": trans_err,
        }
        return errors

    def run_track_all(self) -> None:
        self.generate_initial_camera_poses()
        errors_all: Dict[str, List[float]] = {
            "rot_err": [],
            "trans_err": [],
        }
        for camera_id, target_camera in enumerate(self.target_cameras):
            print("optimizing camera ", camera_id)
            self.optimizer = Adam(
                list(target_camera.parameters()),
                lr=0.01,
                weight_decay=self.optimizer_weight_decay,
            )
            print("initial error")
            errors = self.evaluate_camera_pose(self.cameras[camera_id], target_camera)
            print(errors)
            for step in tqdm(range(100)):
                self.run_track_photometric_step(camera_id, target_camera)
            errors = self.evaluate_camera_pose(self.cameras[camera_id], target_camera)
            print(errors)
            for key in errors:
                errors_all[key].append(errors[key])


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

        #mse: float = float(
        #    torch.mean(torch.square(self.RGB2YUV(render_result["color"]) - self.RGB2YUV(targets["color"]))).item()
        #)
        #psnr = 10 * math.log10(1.0 / mse)
        #self.logger.write(loss_float, psnr, {"color": loss})
        #print("psnr: ", psnr)

        del loss

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.logger.write_batchend()
        self.logger.next()

        return loss_float

    def run_track_reprojection_step(self, camera_id: int, target_camera: Camera) -> float:
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

        #mse: float = float(
        #    torch.mean(torch.square(self.RGB2YUV(render_result["color"]) - self.RGB2YUV(targets["color"]))).item()
        #)
        #psnr = 10 * math.log10(1.0 / mse)
        #self.logger.write(loss_float, psnr, {"color": loss})
        #print("psnr: ", psnr)

        del loss

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.logger.write_batchend()
        self.logger.next()

        return loss_float