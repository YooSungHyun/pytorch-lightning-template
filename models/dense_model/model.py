import torch
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError
from torch import nn
from utils.config_loader import load_config


class CustomNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        config_cls = load_config(args.config_path)
        self.loss_func = MeanSquaredError(compute_on_cpu=self.args.valid_on_cpu)
        # TODO: Write down your network
        self.dense_batch_fc_tanh = nn.Sequential(
            nn.Linear(config_cls.model.input_dense_dim, config_cls.model.output_dense_dim),
            nn.BatchNorm1d(config_cls.model.output_dense_dim),
            nn.Tanh(),
            nn.Linear(config_cls.model.output_dense_dim, (config_cls.model.output_dense_dim // 2)),
            nn.BatchNorm1d((config_cls.model.output_dense_dim // 2)),
            nn.Tanh(),
        )
        self.fc = nn.Linear(config_cls.model.output_dense_dim // 2, 1)

    def forward(self, features):
        outputs = self.dense_batch_fc_tanh(features)
        logits = self.fc(outputs)
        return logits

    def training_step(self, batch, batch_idx):
        features, labels, feature_lengths, label_lengths = batch
        logits = self(features)
        loss = self.loss_func(logits, labels)
        self.log("train_loss", loss, sync_dist=(self.device != "cpu"))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # lightning do sanity eval step first before going training_step. for check your mistake.
        # I always make mistake on validation logic, so this is good
        # If don't use check this url. https://github.com/Lightning-AI/lightning/issues/2295
        features, labels, feature_lengths, label_lengths = batch
        if self.args.valid_on_cpu:
            features = features.cpu()
            labels = labels.cpu()
            feature_lengths = feature_lengths.cpu()
            label_lengths = label_lengths.cpu()
            self.cpu()
        logits = self(features)
        loss = self.loss_func(logits, labels)

        return {"loss": loss}

    def validation_epoch_end(self, validation_step_outputs):
        loss_mean = torch.tensor([x["loss"] for x in validation_step_outputs], device=self.device).mean()

        # sync_dist use follow this url
        # if using torchmetrics -> https://torchmetrics.readthedocs.io/en/stable/
        # if not using torchmetrics -> https://github.com/Lightning-AI/lightning/discussions/6501
        if self.args.valid_on_cpu:
            # if ddp, each machine output must gather. and lightning can gather only on-gpu items
            self.log("val_loss", loss_mean.cuda(), sync_dist=True)
            # model have to training_step on cuda
            self.cuda()
        else:
            self.log("val_loss", loss_mean, sync_dist=(self.device != "cpu"))
        # self.log_dict(metrics, sync_dist=(self.device != "cpu"))

    def predict_step(self, batch, batch_idx):
        features, labels, feature_lengths, label_lengths = batch
        logits = self(features)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [{"params": [p for p in self.parameters()], "name": "OneCycleLR"}],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.args.warmup_ratio,
            epochs=self.trainer.max_epochs,
            final_div_factor=self.args.final_div_factor,
        )
        lr_scheduler = {"interval": "step", "scheduler": scheduler, "name": "AdamW"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
