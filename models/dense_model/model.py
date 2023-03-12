import torch
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError
from torch import nn
from utils.config_loader import load_config


class CustomNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.drop_scheduler = None
        self.drop_rate = args.dropout_p
        config_cls = load_config(args.config_path)
        self.loss_func = MeanSquaredError(compute_on_cpu=args.valid_on_cpu)
        # TODO: Write down your network
        self.dense_batch_fc_tanh = nn.Sequential(
            nn.Linear(config_cls.model.input_dense_dim, config_cls.model.output_dense_dim),
            nn.BatchNorm1d(config_cls.model.output_dense_dim),
            nn.Tanh(),
            nn.Dropout(self.drop_rate),
            nn.Linear(config_cls.model.output_dense_dim, (config_cls.model.output_dense_dim // 2)),
            nn.BatchNorm1d((config_cls.model.output_dense_dim // 2)),
            nn.Tanh(),
            nn.Dropout(self.drop_rate),
        )
        self.fc = nn.Linear(config_cls.model.output_dense_dim // 2, 1)

    def update_dropout(self, drop_rate):
        self.drop_rate = drop_rate
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate

    def forward(self, features):
        outputs = self.dense_batch_fc_tanh(features)
        logits = self.fc(outputs)
        return logits

    def on_train_start(self):
        from models.dense_model.drop_scheduler import drop_scheduler

        self.drop_scheduler = {}
        if self.args.dropout_p > 0.0:
            self.drop_scheduler["do"] = drop_scheduler(
                self.args.dropout_p,
                self.args.max_epochs,
                self.trainer.num_training_batches,
                self.args.cutoff_epoch,
                self.args.drop_mode,
                self.args.drop_schedule,
            )
            print(
                "on_train_start :: Min DO = %.7f, Max DO = %.7f"
                % (min(self.drop_scheduler["do"]), max(self.drop_scheduler["do"]))
            )

    def training_step(self, batch, batch_idx):
        features, labels, feature_lengths, label_lengths = batch
        if "do" in self.drop_scheduler:
            dropout_p = self.drop_scheduler["do"][self.trainer.global_step]
            self.update_dropout(dropout_p)
            self.log("dropout_p", dropout_p, sync_dist=(self.device != "cpu"))
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
