import os
import json
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from models.dense_model.model import CustomNet
from models.dense_model.datamodule import CustomDataModule
from models.rnn_model.model import LSTMModel
from simple_parsing import ArgumentParser
from arguments.training_args import TrainingArguments
from utils.comfy import dataclass_to_namespace


def main(hparams):
    wandb_logger = WandbLogger(project="lightning-template", name="default", save_dir="./")
    pl.seed_everything(hparams.seed)
    os.makedirs(hparams.output_dir, exist_ok=True)
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

    if hparams.accelerator == "cpu" and hparams.valid_on_cpu is True:
        print("If you run on cpu, valid must go on cpu, It set automatically")
        hparams.valid_on_cpu = False
    elif hparams.strategy == "ddp":
        hparams.strategy = DDPStrategy(timeout=timedelta(days=30))
    elif hparams.strategy == "deepspeed_stage_2":
        if hparams.deepspeed_config is not None:
            from pytorch_lightning.strategies import DeepSpeedStrategy

            hparams.strategy = DeepSpeedStrategy(config=hparams.deepspeed_config)
    elif hparams.accelerator != "cpu" and (hparams.strategy is not None and "deepspeed" in hparams.strategy):
        raise NotImplementedError("If you want to another deepspeed option and config, PLZ IMPLEMENT FIRST!!")
    trainer = pl.Trainer.from_argparse_args(hparams)

    if hparams.model_select == "linear":
        datamodule = CustomDataModule(hparams)
        model = CustomNet(hparams)
        wandb_logger.watch(model, log="all")
        trainer.fit(model, datamodule=datamodule)
        """ TODO If use config like dict follow this line
        but, model param is duplicated area between training param and model param
        I want to get training param on run script argument, so I can not use it
        """
        # config_cls = load_config(hparams.config_dir)
        # config = config_to_dict(config_cls)
        # with open(os.path.join(hparams.output_dir, "config.json"), "w") as f:
        # json.dump(config, f, ensure_ascii=False, indent=4)
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
