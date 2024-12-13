from pydantic import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..trainer import ModelForTraining


class MnistConfig(BaseModel):
    num_pixels: int = 784
    hidden_dim: int = 128
    num_labels: int = 10


class MnistModel(nn.Module):
    def __init__(self, config: MnistConfig):
        super().__init__()

        self.config = config

        self.layers = nn.Sequential(
            nn.Linear(config.num_pixels, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_labels),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, pixel_values: torch.Tensor):
        # reshape
        h = pixel_values.view(-1, 784)
        logits = self.layers(h)

        return logits


class MnistModelForTraining(ModelForTraining, nn.Module):
    model: nn.Module

    model_config: MnistConfig
    model_config_class = MnistConfig

    def setup_model(self):
        with self.fabric.rank_zero_first():
            print("Initializing model")
            with self.fabric.init_module():
                model = MnistModel(self.model_config)

        self.fabric.barrier()

        # model = self.fabric.broadcast(model)
        self.model = self.fabric.setup_module(model)

    @torch.no_grad()
    def sanity_check(self):
        with self.fabric.autocast():
            pixel_values = torch.randn(1, self.model_config.num_pixels)
            logits = self.model(self.fabric.to_device(pixel_values))
            assert logits.shape == (1, self.model_config.num_labels)

    def train_step(self, batch):
        self.before_train_step()

        pixel_values, targets = batch

        logits = self.model(pixel_values)
        loss = F.nll_loss(logits, targets)

        self.after_train_step()

        return loss

    def eval_step(self, batch):
        self.before_eval_step()

        pixel_values, targets = batch

        logits = self.model(pixel_values)
        loss = F.nll_loss(logits, targets)

        self.after_eval_step()

        return loss

    def before_load_model(self):
        pass

    def after_load_model(self):
        pass

    def after_train_step(self):
        pass

    def before_eval_step(self):
        pass

    def after_eval_step(self):
        pass

    def before_backward(self):
        pass
