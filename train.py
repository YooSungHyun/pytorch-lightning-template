import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from dense_model import CustomNet
from rnn_model import LSTMModel
from datamodule import CustomDataModule
from simple_parsing import ArgumentParser
from training_args import TrainingArguments
from utils import dataclass_to_namespace


def main(hparams):
    wandb_logger = WandbLogger(project="lightning-template", name="default", save_dir="./")
    pl.seed_everything(hparams.seed)

    hparams.logger = wandb_logger

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.output_dir,
        save_top_k=3,
        mode="min",
        monitor="val_loss",
        filename="lightning-template-{epoch:02d}-{val_loss:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    hparams.callbacks = [checkpoint_callback, lr_monitor]

    if hparams.strategy == "ddp":
        hparams.strategy = DDPStrategy(timeout=timedelta(days=30))

    trainer = pl.Trainer.from_argparse_args(hparams)

    if hparams.model_select == "linear":
        ncf_datamodule = CustomDataModule(hparams)
        model = CustomNet(hparams)
        wandb_logger.watch(model, log="all")
        trainer.fit(model, datamodule=ncf_datamodule)
    else:
        model = LSTMModel(hparams)
        wandb_logger.watch(model, log="all")
        trainer.fit(model)
    # TODO If finetuning follow this line
    # PreTrainedLightningModule.load_state_dict(
    #     torch.load(
    #         "",
    #         map_location="cuda",
    #     ),
    #     strict=False,
    # )
    checkpoint_callback.best_model_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_arguments(TrainingArguments, dest="training_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "training_args")
    main(args)
