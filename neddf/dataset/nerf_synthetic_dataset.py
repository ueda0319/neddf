import json
from pathlib import Path
from typing import Dict, Final, List

import cv2
import numpy as np
from neddf.dataset.base_dataset import BaseDataset
from numpy import ndarray
from scipy.spatial.transform import Rotation


class NeRFSyntheticDataset(BaseDataset):
    """Dataset class for nerf_synthetic_dataset.

    This is Dataset class for nerf_synthetic_dataset.
    (https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

    Attributes:
        camera_calib_params (ndarray[4, float]): camera intrisic parameter [fx, fy, cx, cy]
        camera_params (ndarray[bs, 6, float]): camera pose parameter [rx, ry, rz, px, py, pz]
        rgb_images (ndarray[bs, h, w, 3, uint8]): rgb images
        mask_images (ndarray[bs, h, w, uint8]): mask images
    """

    def load_data(self) -> None:
        # Load dataset design file
        transform_path: Final[Path] = self.dataset_dir / "transforms_{}.json".format(
            self.data_split
        )
        with open(transform_path.as_posix()) as f:
            transform_data = json.load(f)

        img_0_path: Final[Path] = self.dataset_dir / (
            transform_data["frames"][0]["file_path"] + ".png"
        )
        img_0: ndarray = cv2.imread(img_0_path.as_posix(), cv2.IMREAD_UNCHANGED)
        h: Final[int] = img_0.shape[0]
        w: Final[int] = img_0.shape[1]
        camera_angle_x: Final[float] = float(transform_data["camera_angle_x"])
        focal: Final[float] = 0.5 * w / np.tan(0.5 * camera_angle_x)

        rgb_images: List[ndarray] = []
        mask_images: List[ndarray] = []
        camera_params: List[ndarray] = []
        for frame in transform_data["frames"]:
            # Get camera pose
            transform_matrix: ndarray = np.array(frame["transform_matrix"])
            camera_param: ndarray = np.zeros(6, np.float32)
            camera_param[:3] = Rotation.from_matrix(
                transform_matrix[:3, :3]
            ).as_rotvec()
            camera_param[3:] = transform_matrix[:3, 3]
            camera_params.append(camera_param)

            # Get image
            img_path: Path = self.dataset_dir / (frame["file_path"] + ".png")
            img: ndarray = cv2.imread(img_path.as_posix(), cv2.IMREAD_UNCHANGED)
            rgb_images.append(img[:, :, :3])
            mask_images.append(img[:, :, 3])

        self.camera_calib_params: ndarray = np.array([focal, focal, 0.5 * w, 0.5 * h])
        self.camera_params: ndarray = np.stack(camera_params, 0)
        self.rgb_images: ndarray = np.stack(rgb_images, 0)
        self.mask_images: ndarray = np.stack(mask_images, 0)

    def __getitem__(self, item: int) -> Dict[str, ndarray]:
        return {
            "camera_calib_params": self.camera_calib_params,
            "camera_params": self.camera_params[item, :],
            "rgb_images": self.rgb_images[item, :, :, :],
            "mask_images": self.mask_images[item, :, :],
        }
