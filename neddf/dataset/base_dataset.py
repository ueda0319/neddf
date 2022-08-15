from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np
from numpy import ndarray
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Abstract base class for dataset.

    Attributes:
        camera_calib_params (ndarray[4, float]): camera intrisic parameter [fx, fy, cx, cy]
        camera_params (ndarray[bs, 6, float]): camera pose parameter [rx, ry, rz, px, py, pz]
        rgb_images (ndarray[bs, h, w, 3, uint8]): rgb images
        mask_images (ndarray[bs, h, w, uint8]): mask images
        depth_images (ndarray[bs, h, w, uint8]): depth images
        use_depth (bool): True when dataset includes depth datas
        use_mask (bool): True when dataset includes mask datas
    """

    def __init__(
        self,
        dataset_dir: str,
        data_split: str,
        use_depth: bool = False,
        use_mask: bool = False,
    ) -> None:
        """Initializer

        This method initialize common attributes.

        Args:
            dataset_dir (str): Path to dataset directory
            data_split (str): Dataset split type
                Usually takes one of ['train', 'test', 'valid'].

        """
        self.dataset_dir: Path = Path(dataset_dir)
        self.data_split: str = data_split
        self.camera_calib_params: ndarray = np.zeros(4)
        self.camera_params: ndarray = np.zeros((1, 6))
        self.rgb_images: ndarray = np.zeros(0)
        self.mask_images: ndarray = np.zeros(0)
        self.depth_images: ndarray = np.zeros(0)
        self.use_depth: bool = use_depth
        self.use_mask: bool = use_mask

        self.load_data()

    @abstractmethod
    def load_data(self) -> None:
        """Abstract method for Load Dataset

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
        raise NotImplementedError()

    def __len__(self) -> int:
        """Special method called in len(self)

        Get count of items
        The implementation is needed in torch.utils.data.Dataset
        """
        return self.rgb_images.shape[0]

    @property
    def image_width(self) -> int:
        """int: width of images in the dataset"""
        return self.rgb_images.shape[2]

    @property
    def image_height(self) -> int:
        """int: height of images in the dataset"""
        return self.rgb_images.shape[1]
