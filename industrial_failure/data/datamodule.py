import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from omegaconf import DictConfig

from industrial_failure.data.download import download_data
from industrial_failure.utils.preprocessing import preprocess_features


class EquipmentDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        df = download_data()
        features, _ = preprocess_features(df)
        target = df[self.cfg.data.target_col].values

        self.input_dim = features.shape[1]

        self.train_dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
        )
