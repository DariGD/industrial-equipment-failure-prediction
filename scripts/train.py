import hydra
from omegaconf import DictConfig

from industrial_failure.training.train_baseline import train_baseline
from industrial_failure.training.train_nn import train_nn


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.model.name == "logistic_regression":
        train_baseline(cfg)
    elif cfg.model.name == "neural_net":
        train_nn(cfg)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.name}")


if __name__ == "__main__":
    main()
