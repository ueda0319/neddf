from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from melon.trainer import NeRFTrainer


class TestNeRFTrainer:
    def test_trainer(self):
        conf_dir = Path.cwd() / "config"
        assert conf_dir.is_dir()
        GlobalHydra.instance().clear()
        hydra.initialize_config_dir(config_dir=conf_dir.as_posix())
        cfg: DictConfig = hydra.compose(
            config_name="default", overrides=["dataset=test", "trainer=test"]
        )
        trainer: NeRFTrainer = NeRFTrainer(cfg)
        trainer.run_train_step(0)
