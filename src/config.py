from typing import Literal

import yaml
from pathlib import Path

from pydantic import BaseModel


class ModelConfig(BaseModel):
    pass


class DatasetConfig(BaseModel):
    train: str
    val: str


class LoggingConfig(BaseModel):
    provider: str | None = None
    project_name: str | None = None
    run_name: str | None = None


class SaveConfig(BaseModel):
    # path to save the model
    directory: str


class TrainConfig(BaseModel):
    model: ModelConfig
    dataset: DatasetConfig
    save: SaveConfig
    logging: LoggingConfig | None = None

    optimizer: str = "torch.optim.AdamW"
    optimizer_args: dict = {}

    scheduler: str = "torch.optim.lr_scheduler.ConstantLR"
    scheduler_args: dict = {}

    trainer_args: dict = {}
    dataloaders_args: dict = {}

    torch_compile: bool = False
    fp32_matmul_precision: str | None = None

    def to_dict(self) -> dict:
        return self.model_dump()

    def save_to(self, dir: Path | str, filename: str = "config.yaml"):
        if isinstance(dir, str):
            dir = Path(dir)

        dir.mkdir(parents=True, exist_ok=True)
        with open(dir / filename, "w") as f:
            yaml.dump(self.to_dict(), f)

    @staticmethod
    def from_config_file(path: str) -> "TrainConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        return TrainConfig.model_validate(config)
