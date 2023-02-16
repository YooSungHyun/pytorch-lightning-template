import os
import torch
import pytorch_lightning as pl
from argparse import Namespace
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx], len(self.x[idx]), len(self.y[idx]))


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.data_dir = args.data_dir
        self.ratio = args.ratio
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size
        self.num_workers = args.num_workers

    def prepare_data(self):
        if not os.path.isfile("./valid.pt"):
            features = torch.randn(2000, 512)
            labels = torch.randn(2000, 1)
            train_x, valid_x, train_y, valid_y = train_test_split(features, labels, test_size=self.ratio)
            train_datasets = CustomDataset(train_x, train_y)
            valid_datasets = CustomDataset(valid_x, valid_y)
            torch.save(train_datasets, "./train.pt")
            torch.save(valid_datasets, "./valid.pt")
        else:
            pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_datasets = torch.load("./train.pt")
            self.valid_datasets = torch.load("./valid.pt")
        if stage == "test":
            pass

    def train_dataloader(self):
        # TODO: If you want to use custom sampler and loader follow like this
        # from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
        # train_sampler = DistributedLengthGroupedSampler(
        # batch_size=self.per_device_eval_batch_size,
        # dataset=self.train_datasets,
        # model_input_name="features",
        # lengths=self.train_datasets["feature_lenghths"],
        # )
        # return CustomDataLoader(
        #     dataset=self.train_datasets,
        #     batch_size=self.per_device_train_batch_size,
        #     sampler=train_sampler,
        #     num_workers=self.num_workers,
        #     pin_memory=True,
        # )

        # TODO: If you want to use default loader
        return DataLoader(
            dataset=self.train_datasets,
            batch_size=self.per_device_train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_datasets,
            batch_size=self.per_device_train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        pass
