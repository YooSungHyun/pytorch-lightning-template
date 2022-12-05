import argparse
import torch
import pytorch_lightning as pl
from dense_model import CustomNet
from rnn_model import LSTMModel
from utils import str2bool


def main(hparams):
    pl.seed_everything(hparams.seed)

    if hparams.model_select == "linear":
        model = CustomNet.load_from_checkpoint(hparams.model_path, args=hparams)
        features = torch.randn(1, 512)
    else:
        model = LSTMModel.load_from_checkpoint(hparams.model_path, args=hparams)
        features = torch.randn(200, 1)
    model.eval()
    with torch.no_grad():
        logits = model(features)
    print(logits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", default=None, type=int, help="all seed")
    parser.add_argument("--local_rank", type=int, help="ddp local rank")
    parser.add_argument("--model_path", type=str, help="model output path")
    parser.add_argument("--input_dense_dim", default=512, type=int, help="input network dimension")
    parser.add_argument("--output_dense_dim", default=256, type=int, help="output network dimension")
    parser.add_argument("--model_select", default="linear", type=str, help="linear or rnn")
    parser.add_argument("--truncated_bptt_steps", default=1, type=int, help="TBPTT step size")
    parser.add_argument(
        "--valid_on_cpu", default=False, type=str2bool, help="If you want to run validation_step on cpu -> true"
    )
    args = parser.parse_args()
    main(args)
