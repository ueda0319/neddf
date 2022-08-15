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
        """Load Dataset

        Load images and camera poses from Dataset

        Note:
            This method is called during initialization.
            `dataset_dir` and `dataset_split` are available in this method.
            On inheritance, register the values in
                `camera_calib_param`, `camera_params`, `rgb_images`.
            (And register `mask_images` and `depth_images` if the dataset include them.)
        """
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
            if self.use_mask:
                img: ndarray = cv2.imread(img_path.as_posix(), cv2.IMREAD_UNCHANGED)
                rgb = (
                    (1.0 / 256)
                    * img[:, :, 3, None].astype(np.float32)
                    * img[:, :, :3].astype(np.float32)
                )
                rgb_images.append(rgb)
                mask_images.append(img[:, :, 3])
            else:
                img = cv2.imread(img_path.as_posix(), cv2.IMREAD_UNCHANGED)[:, :, :3]
                rgb_images.append(img.astype(np.float32))
                mask_images.append(255 * np.ones_like(img[:, :, 0]))

        self.camera_calib_params: ndarray = np.array([focal, focal, 0.5 * w, 0.5 * h])
        self.camera_params: ndarray = np.stack(camera_params, 0)
        self.rgb_images: ndarray = np.stack(rgb_images, 0)
        self.mask_images: ndarray = np.stack(mask_images, 0)

    def __getitem__(self, item: int) -> Dict[str, ndarray]:
        """Special method called in self[item]

        Get item in selected index
        The implementation is needed in torch.utils.data.Dataset

        Args:
            item (int): index of item

        Returns:
            Dict[str, ndarray]: dictionary of each item
                Key takes `camera_calib_param`, `camera_params`, `rgb_images` and etc.
        """
        return {
            "camera_calib_params": self.camera_calib_params,
            "camera_params": self.camera_params[item, :],
            "rgb_images": self.rgb_images[item, :, :, :],
            "mask_images": self.mask_images[item, :, :],
        }
