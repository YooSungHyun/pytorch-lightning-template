import argparse
from utils import str2bool
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from model import CustomNet
from datamodule import CustomDataModule


def main(hparams):
    wandb_logger = WandbLogger(project="lightning-template", name="default", save_dir="./")
    pl.seed_everything(hparams.seed)

    ncf_datamodule = CustomDataModule(hparams)
    model = CustomNet(hparams)
    # TODO If finetuning follow this line
    # PreTrainedLightningModule.load_state_dict(
    #     torch.load(
    #         "",
    #         map_location="cuda",
    #     ),
    #     strict=False,
    # )
    wandb_logger.watch(model, log="all")
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
    trainer.fit(model, datamodule=ncf_datamodule)
    checkpoint_callback.best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", default=None, type=int, help="all seed")
    parser.add_argument("--local_rank", type=int, help="ddp local rank")
    parser.add_argument("--data_dir", type=str, help="target pytorch lightning data dirs")
    parser.add_argument("--ratio", type=float, help="train/valid split ratio")
    parser.add_argument("--output_dir", type=str, help="model output path")
    parser.add_argument("--num_proc", type=int, default=None, help="how many proc map?")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument(
        "--warmup_ratio", default=0.2, type=float, help="learning rate scheduler warmup ratio per EPOCH"
    )
    parser.add_argument("--max_lr", default=0.01, type=float, help="lr_scheduler max learning rate")
    parser.add_argument("--div_factor", default=25, type=int, help="initial_lr = max_lr/div_factor")
    parser.add_argument(
        "--final_div_factor", default=1e4, type=int, help="(max_lr/div_factor)*final_div_factor is final lr"
    )
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="weigth decay")
    parser.add_argument(
        "--per_device_train_batch_size", default=1, type=int, help="The batch size per GPU/TPU core/CPU for training."
    )
    parser.add_argument(
        "--per_device_eval_batch_size", default=1, type=int, help="The batch size per GPU/TPU core/CPU for evaluation."
    )
    parser.add_argument("--input_dense_dim", default=512, type=int, help="input network dimension")
    parser.add_argument("--output_dense_dim", default=256, type=int, help="output network dimension")
    parser.add_argument(
        "--valid_on_cpu", default=False, type=str2bool, help="If you want to run validation_step on cpu -> true"
    )
    args = parser.parse_args()
    main(args)
