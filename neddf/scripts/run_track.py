from argparse import ArgumentParser
from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from neddf.trainer import BaseTrainer
from omegaconf import DictConfig


def main() -> None:
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "output_dir",
        type=Path,
        help="directory path where models and render are located",
    )
    parser.add_argument("--epoch", type=int, default=2000, help="epoch number of model")
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    # reconstruct config
    conf_dir: Path = output_dir / ".hydra"
    assert conf_dir.is_dir()
    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=conf_dir.as_posix())
    cfg: DictConfig = hydra.compose(
        config_name="config", overrides=["dataset.data_split=test", "trainer._target_=neddf.trainer.NeDDFTracker"]
    )
    trainer: BaseTrainer = hydra.utils.instantiate(
        cfg.trainer,
        global_config=cfg,
        _recursive_=False,
    )

    # load model path
    model_path: Path = output_dir / "models/model_{:05}.pth".format(args.epoch)
    # assert model_path.exists()
    trainer.load_pretrained_model(model_path)

    # render all
    save_dir: Path = args.output_dir / "eval"
    save_dir.mkdir(exist_ok=True)
    trainer.run_track_all()


if __name__ == "__main__":
    main()
