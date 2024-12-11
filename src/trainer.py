import torch
import torch.nn as nn
import torch.optim.optimizer as optim
import torch.nn.functional as F
import torch.utils.data as data
from abc import ABC, abstractmethod


class ModelForTraining(ABC):
    @abstractmethod
    def train_step(self, batch):
        pass

    @abstractmethod
    def eval_step(self, batch):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def before_forward(self):
        pass

    @abstractmethod
    def after_forward(self):
        pass

    @abstractmethod
    def before_backward(self):
        pass

    @abstractmethod
    def after_backward(self):
        pass


class Trainer:
    def __init__(
        self,
        model: ModelForTraining,
        train_dataloader: data.DataLoader,
        eval_dataloader: data.DataLoader | None = None,
        optimizer: optim.Optimizer | None = None,
    ) -> None:
        pass

    def training_loop(self):
        pass
