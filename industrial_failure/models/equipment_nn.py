import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )
        pt = torch.exp(-bce_loss)
        loss = ((1 - pt) ** self.gamma * bce_loss).mean()
        return loss


class EquipmentNN(pl.LightningModule):
    def __init__(self, cfg, input_dim=None):
        super().__init__()
        self.save_hyperparameters(cfg)
        hidden_dims = cfg.model.hidden_dims
        dropout = cfg.model.dropout
        self.input_dim = input_dim if input_dim is not None else cfg.model.input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )
        self.criterion = FocalLoss(gamma=cfg.model.gamma)
        self.accuracy = Accuracy(task="binary")
        self.auroc = AUROC(task="binary")
        self.lr = cfg.model.lr

    def forward(self, x):
        return self.model(x).squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits) > 0.5
        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(preds, y.int()))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits) > 0.5
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy(preds, y.int()), prog_bar=True)
        self.log(
            "val_roc_auc", self.auroc(torch.sigmoid(logits), y.int()), prog_bar=True
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
