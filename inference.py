import argparse
import torch
import pytorch_lightning as pl
from model import CustomNet


def main(hparams):
    pl.seed_everything(hparams.seed)

    model = CustomNet.load_from_checkpoint(hparams.model_path, args=hparams)
    model.eval()
    features = torch.randn(1, 512)
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
    args = parser.parse_args()
    main(args)
