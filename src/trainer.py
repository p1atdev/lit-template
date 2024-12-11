from abc import ABC, abstractmethod


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from lightning.fabric import Fabric

from .optimizer import get_optimizer
from .scheduler import get_scheduler, NothingScheduler
from .config import OptimizerConfig, SchedulerConfig, TrainConfig
from .saving import ModelSavingStrategy, get_saving_callback


class ModelForTraining(ABC):
    fabric: Fabric
    config: TrainConfig

    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler

    def __init__(
        self,
        fabric: Fabric,
        config: TrainConfig,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.config = config
        self.fabric = fabric

    @abstractmethod
    def setup_model(self):
        pass

    def setup_optimizer(self):
        optimizer = get_optimizer(
            self.config.optimizer.name,
            self.model.parameters(),
            **self.config.optimizer.args,
        )
        if (scheduler_config := self.config.scheduler) is not None:
            scheduler = get_scheduler(
                optimizer,
                scheduler_config.name,
                **scheduler_config.args,
            )
        else:
            scheduler = NothingScheduler(optimizer)

        self.optimizer = self.fabric.setup_optimizers(
            optimizer,
        )  # type: ignore  # Fabric's setup_optimizers method may not be recognized by type checkers
        self.scheduler = scheduler

    @abstractmethod
    def train_step(self, batch) -> torch.Tensor:
        pass

    @abstractmethod
    def eval_step(self, batch) -> torch.Tensor:
        pass

    def backward(self, loss: torch.Tensor):
        self.before_backward()

        self.fabric.backward(loss)

        self.after_backward()

    @abstractmethod
    def before_load_model(self):
        pass

    @abstractmethod
    def after_load_model(self):
        pass

    def before_train_step(self):
        self.optimizer.zero_grad()

    @abstractmethod
    def after_train_step(self):
        pass

    @abstractmethod
    def before_eval_step(self):
        pass

    @abstractmethod
    def after_eval_step(self):
        pass

    @abstractmethod
    def before_backward(self):
        pass

    def after_backward(self):
        self.optimizer.step()
        self.scheduler.step()

    def before_train_epoch(self):
        self.model.train()
        # if the optimizer has train(), call it
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()  # type: ignore  # Some optimizers might not have a train method

    def after_train_epoch(self):
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()  # type: ignore  # Some optimizers might not have an eval method

    def before_eval_epoch(self):
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()  # type: ignore

    def after_eval_epoch(self):
        pass

    def before_save_model(self):
        pass

    def after_save_model(self):
        pass


class Trainer:
    model: ModelForTraining

    def __init__(
        self,
        config: TrainConfig,
        train_dataloader: data.DataLoader,
        eval_dataloader: data.DataLoader | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.seed = seed

        self.saving_strategy = ModelSavingStrategy.from_config(
            config=self.config.saving.strategy,
            steps_per_epoch=len(self.train_dataloader),
            total_epochs=self.config.num_train_epochs,
        )
        self.saving_callbacks = self.get_saving_callbacks()

        self.fabric = Fabric()

    def set_model_class(self, model_cls, *args, **kwargs):
        self.model = model_cls(self.fabric, self.config, *args, **kwargs)

    def get_saving_callbacks(self):
        return [
            get_saving_callback(callback) for callback in self.config.saving.callbacks
        ]

    def before_train(self):
        self.fabric.seed_everything(self.seed)

        self.model.setup_optimizer()

    def after_train(self):
        pass

    def training_loop(self):
        current_step = 0
        total_epochs = self.config.num_train_epochs

        for epoch in range(total_epochs):
            self.model.before_train_epoch()

            for batch in self.train_dataloader:
                current_step += 1
                self.model.before_train_step()

                loss = self.model.train_step(batch)
                self.model.backward(loss)

                self.model.after_train_step()
                self.call_saving_callbacks(epoch, current_step)

            self.model.after_train_epoch()
            self.call_saving_callbacks(epoch, current_step)

            if self.eval_dataloader is not None:
                self.model.before_eval_epoch()

                for batch in self.eval_dataloader:
                    self.model.before_eval_step()

                    loss = self.model.eval_step(batch)

                    self.model.after_eval_step()

                self.model.after_eval_epoch()

    def call_saving_callbacks(self, epoch: int, steps: int):
        if self.saving_strategy.should_save(epoch, steps):
            self.model.before_save_model()

            for callback in self.saving_callbacks:
                callback.save(self.model.model, epoch, steps)

            self.model.after_save_model()

    def train(self):
        self.before_train()

        self.model.setup_model()
        self.model.setup_optimizer()

        self.training_loop()

        self.after_train()
