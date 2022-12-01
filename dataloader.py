import torch


class CustomDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(CustomDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        features = [s["features"] for s in batch]
        feature_lengths = [s["features"].size(0) for s in batch]
        labels = [s["label"] for s in batch]
        label_lengths = [len(s["label"]) for s in batch]

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        feature_lengths = torch.IntTensor(feature_lengths)
        label_lengths = torch.IntTensor(label_lengths)

        return features, labels, feature_lengths, label_lengths
