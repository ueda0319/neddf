from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


class TestNeRFSyntheticDataset:
    @classmethod
    def setup_class(cls):
        conf_dir = Path.cwd() / "config"
        assert conf_dir.is_dir()
        GlobalHydra.instance().clear()
        hydra.initialize_config_dir(config_dir=conf_dir.as_posix())
        cls.cfg: DictConfig = hydra.compose(
            config_name="default", overrides=["dataset=test"]
        )

    @classmethod
    def teardown_class(cls):
        del cls.cfg

    def test_dataset(self):
        dataset = hydra.utils.instantiate(self.cfg.dataset)
        h = dataset.image_height
        w = dataset.image_width
        assert dataset[0]["camera_calib_params"].shape == (4,)
        assert dataset[0]["camera_params"].shape == (6,)
        assert dataset[0]["rgb_images"].shape == (h, w, 3)
        assert dataset[0]["mask_images"].shape == (h, w)
