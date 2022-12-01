import torch
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError
from torch import nn


class CustomNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        # TODO: If you want to learn eval metric on cpu. just use compute_on_cpu=True
        # and follow validation_step
        self.loss_func = MeanSquaredError(compute_on_cpu=False)
        # TODO: Write down your network
        self.dense_batch_fc_tanh = nn.Sequential(
            nn.Linear(args.input_dense_dim, args.output_dense_dim),
            nn.BatchNorm1d(args.output_dense_dim),
            nn.Tanh(),
            nn.Linear(args.output_dense_dim, (args.output_dense_dim // 2)),
            nn.BatchNorm1d((args.output_dense_dim // 2)),
            nn.Tanh(),
        )
        self.fc = nn.Linear(args.output_dense_dim // 2, 1)

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
        """validation_step
        if you want to run validation_step on cpu, follow this,
        1. All variable move to cpu
            features.cpu()
            labels.cpu()
            feature_lengths.cpu()
            label_lengths.cpu()
        2. model move to cpu
            self.cpu()
        3. eval step going

        Args:
            batch (_type_): validation datasets batch
            batch_idx (_type_): batch's index

        """
        features, labels, feature_lengths, label_lengths = batch
        logits = self(features)
        loss = self.loss_func(logits, labels)

        return {"loss": loss}

    def validation_epoch_end(self, validation_step_outputs):
        """validation_epoch_end
        if you want to run validation_step on cpu, follow this,
        1. your validation_step_outputs is already on cpu
        2. if ddp, each machine output must gather. and lightning can gather only on-gpu items
            self.log("val_loss", loss_mean.cuda(), sync_dist=True)
            self.cuda() -> model have to training_step on cuda

        Args:
            batch (_type_): validation datasets batch
            batch_idx (_type_): batch's index

        """
        loss_mean = torch.tensor([x["loss"] for x in validation_step_outputs], device=self.device).mean()
        self.log("val_loss", loss_mean, sync_dist=(self.device != "cpu"))
        # self.log_dict(metrics, sync_dist=(self.device != "cpu"))

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
