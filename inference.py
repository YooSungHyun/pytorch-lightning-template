import torch
import pytorch_lightning as pl
from models.dense_model.model import CustomNet
from models.rnn_model.model import LSTMModel
from utils.compy import dataclass_to_namespace
from arguments.inference_args import InferenceArguments
from simple_parsing import ArgumentParser


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
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_arguments(InferenceArguments, dest="inference_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "inference_args")
    main(args)
