import os
import random
from pathlib import Path
from typing import Final

import hydra
import numpy as np
import torch
from neddf.trainer import BaseTrainer
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="default")
def main(cfg: DictConfig) -> None:
    cwd: Final[Path] = Path(hydra.utils.get_original_cwd())
    cfg.dataset.dataset_dir = str(cwd / cfg.dataset.dataset_dir)
    trainer: BaseTrainer = hydra.utils.instantiate(
        cfg.trainer,
        global_config=cfg,
        _recursive_=False,
    )
    trainer.run_train()


if __name__ == "__main__":
    # Set current directory for run from python (not poetry)
    if "neddf/neddf" in os.getcwd():
        os.chdir("..")

    # Fix seed
    seed: Final[int] = 3408
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)
    main()
