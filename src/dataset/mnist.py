import torch
import torch.utils.data as data

from datasets import load_dataset, Dataset, DatasetDict

from .util import DatasetConfig


class MnistDatasetConfig(DatasetConfig):
    repo_id: str = "ylecun/mnist"

    def get_dataset(self):
        ds = load_dataset(self.repo_id)
        assert isinstance(ds, DatasetDict)

        return MnistDataset(ds["train"]), MnistDataset(ds["test"])


class MnistDataset(data.Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        label = self.dataset[idx]["label"]

        # pil to tensor
        image = torch.tensor(image, dtype=torch.float32).view(-1)
        label = torch.tensor(label, dtype=torch.int64)

        return image, label
