import mlflow
from omegaconf import DictConfig
import hydra
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from industrial_failure.data.datamodule import EquipmentDataModule
from industrial_failure.models.equipment_nn import EquipmentNN


def train_nn(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.logging.mlflow_uri)
    mlflow.set_experiment(cfg.logging.experiment_name)

    data_module = EquipmentDataModule(cfg)
    data_module.setup()

    input_dim = data_module.input_dim

    model = EquipmentNN(cfg, input_dim=input_dim)

    logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.mlflow_uri,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=logger,
        accelerator="auto",
        deterministic=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=data_module)

    model_dir = Path("industrial_failure/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "equipment_nn.ckpt"
    trainer.save_checkpoint(ckpt_path)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="nn",
)
def main(cfg: DictConfig):
    train_nn(cfg)


if __name__ == "__main__":
    main()
