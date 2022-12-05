import torch
from torchmetrics import MeanSquaredError
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule


class LSTMModel(LightningModule):
    """LSTM sequence-to-sequence model for testing TBPTT with automatic optimization."""

    def __init__(self, truncated_bptt_steps, valid_on_cpu):
        super().__init__()
        self.valid_on_cpu = valid_on_cpu
        self.input_size = 1
        self.hidden_size = 8
        self.lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_size * 2, 1)
        self.loss_func = MeanSquaredError(compute_on_cpu=self.valid_on_cpu)
        self.truncated_bptt_steps = truncated_bptt_steps
        self.automatic_optimization = True

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx, hiddens):
        # batch_idx: Original step indices, Not TBPTT index (1 step == 1 batch)
        # hiddens: TBPTT use backwards each sequence using this data

        # On tbptt, backpropagation is used CHUNK by long sequence. when if using 200 sequence and 100 step chunk,
        # training_step is needed 2 step for 1 batch (1 step: 0~99, 2 step: 100~199)
        # very cleverly, we just using hiddens parameter, lightning's tbptt not connected new batch's hiddens to past one
        x, y = batch
        if hiddens is not None:
            hiddens1, hiddens2 = hiddens
        else:
            hiddens1 = None
            hiddens2 = None
        lstm_last, hiddens1 = self.lstm(x, hiddens1)
        lstm2_last, hiddens2 = self.lstm2(x, hiddens2)
        concat_lstm = torch.concat([lstm_last, lstm2_last], dim=-1)
        logits = self.linear(concat_lstm)
        loss = self.loss_func(logits, y)
        self.log("train_loss", loss, sync_dist=(self.device != "cpu"))
        # look this discussion for tbptt experiment (https://github.com/Lightning-AI/lightning/discussions/15643)
        return {"loss": loss, "hiddens": (hiddens1, hiddens2)}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.valid_on_cpu:
            x = x.cpu()
            y = y.cpu()
            self.cpu()
        lstm_last, _ = self.lstm(x)
        lstm2_last, _ = self.lstm2(x)
        concat_lstm = torch.concat([lstm_last, lstm2_last], dim=-1)
        logits = self.linear(concat_lstm)
        return {"pred": logits, "labels": y}

    def validation_epoch_end(self, validation_step_outputs):
        for out in validation_step_outputs:
            loss = self.loss_func(out["pred"], out["labels"])
        if self.valid_on_cpu:
            # if ddp, each machine output must gather. and lightning can gather only on-gpu items
            self.log("val_loss", loss.cuda(), sync_dist=True)
            # model have to training_step on cuda
            self.cuda()
        else:
            self.log("val_loss", loss, sync_dist=(self.device != "cpu"))

    def train_dataloader(self):
        dataset = TensorDataset(torch.rand(50, 200, self.input_size), torch.rand(50, 200, self.input_size))
        return DataLoader(dataset=dataset, batch_size=4)

    def val_dataloader(self):
        dataset = TensorDataset(torch.rand(50, 200, self.input_size), torch.rand(50, 200, self.input_size))
        return DataLoader(dataset=dataset, batch_size=4)