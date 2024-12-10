import importlib
import torch
from typing import Any, Iterable

from transformers.optimization import get_scheduler as hf_get_scheduler
from transformers.trainer_utils import SchedulerType

from .config import TrainConfig


def get_optimizer(config: TrainConfig, params: Iterable[Any]) -> torch.optim.Optimizer:
    """
    Get the optimizer from the optimizer name and the arguments
    """

    if "." not in config.optimizer:
        module_name = "torch.optim"
        optimizer_name = config.optimizer
    else:
        module_name, optimizer_name = config.optimizer.rsplit(".", 1)
        print(f"Using custom optimizer {optimizer_name} from {module_name}")

    try:
        module = importlib.import_module(module_name)
        optimizer_class = getattr(module, optimizer_name)

        lr = config.optimizer_args.pop("lr", None)
        assert lr is not None, "Learning rate must be provided"
        print(f"Using lr={lr} as the learning rate")

        return optimizer_class(params, lr=lr, **config.optimizer_args)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Optimizer {optimizer_name} not found in module {module_name}"
        ) from e


def get_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainConfig
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get the scheduler from the scheduler name and the arguments
    """

    # TODO: get total steps

    try:
        # Try to use the transformers scheduler
        scheduler_type = SchedulerType(config.scheduler)
        return hf_get_scheduler(scheduler_type, optimizer, **config.scheduler_args)
    except ValueError:
        pass

    if "." not in config.scheduler:
        module_name = "torch.optim.lr_scheduler"
        scheduler_name = config.scheduler
    else:
        module_name, scheduler_name = config.scheduler.rsplit(".", 1)
        print(f"Using custom scheduler {scheduler_name} from {module_name}")

    try:
        module = importlib.import_module(module_name)
        scheduler_class = getattr(module, scheduler_name)
        return scheduler_class(optimizer, **config.scheduler_args)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Scheduler {scheduler_name} not found in module {module_name}"
        ) from e
