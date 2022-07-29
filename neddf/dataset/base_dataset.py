from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Abstract base class for dataset.

    Attributes:
        camera_calib_params (np.ndarray[4, float]): camera intrisic parameter [fx, fy, cx, cy]
        camera_params (np.ndarray[bs, 6, float]): camera pose parameter [rx, ry, rz, px, py, pz]
        rgb_images (np.ndarray[bs, h, w, 3, uint8]): rgb images
        mask_images (np.ndarray[bs, h, w, uint8]): mask images
        depth_images (np.ndarray[bs, h, w, uint8]): depth images
    """

    def __init__(
        self,
        dataset_dir: str,
        data_split: str,
    ) -> None:
        self.dataset_dir: Path = Path(dataset_dir)
        self.data_split: str = data_split
        self.camera_calib_params: np.ndarray = np.zeros(4)
        self.camera_params: np.ndarray = np.zeros((1, 6))
        self.rgb_images: np.ndarray = np.zeros(0)
        self.mask_images: np.ndarray = np.zeros(0)
        self.depth_images: np.ndarray = np.zeros(0)

        self.load_data()

    @abstractmethod
    def load_data(self) -> None:
        """Abstract method for load dataset

        Load images and camera poses from Dataset

        Note:
            This method is called during initialization.
            `dataset_dir` and `dataset_split` are available in this method.
            On inheritance, register the values in
                `camera_calib_param`, `camera_params`, `rgb_images`.
            (And register `mask_images` and `depth_images` if the dataset include them.)

        """
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, item: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.rgb_images.shape[0]

    @property
    def image_width(self) -> int:
        return self.rgb_images.shape[2]

    @property
    def image_height(self) -> int:
        return self.rgb_images.shape[1]
