import argparse

from src.models.mnist import MnistModelForTraining
from src.trainer import Trainer
from src.config import TrainConfig


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = prepare_args()

    config = TrainConfig.from_config_file(args.config)

    trainer = Trainer(config, train_dataloader=range(100))
    trainer.setup_model(MnistModelForTraining)


if __name__ == "__main__":
    main()
