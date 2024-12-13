import torch
import torch.utils.data as data
import torchvision.transforms.functional as F

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
        image = self.dataset[idx]["image"]  # PngImageFile
        label = self.dataset[idx]["label"]

        # pil to tensor
        image_tensor = F.pil_to_tensor(image)
        label_tensor = torch.LongTensor([label])

        return image_tensor, label_tensor
