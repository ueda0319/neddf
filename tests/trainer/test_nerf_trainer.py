from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from neddf.trainer import BaseTrainer


class TestNeRFTrainer:
    def test_trainer(self):
        conf_dir = Path.cwd() / "config"
        assert conf_dir.is_dir()
        GlobalHydra.instance().clear()
        hydra.initialize_config_dir(
            version_base=None, 
            config_dir=conf_dir.as_posix()
        )
        cfg: DictConfig = hydra.compose(
            config_name="default", overrides=["dataset=test", "trainer=test"]
        )
        trainer: BaseTrainer = hydra.utils.instantiate(
            cfg.trainer,
            global_config=cfg,
            _recursive_=False,
        )
        trainer.run_train_step(0)
