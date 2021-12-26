import os
from pathlib import Path

import click
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import yaml
from dotenv import load_dotenv
from easydict import EasyDict
from optuna import Trial
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold

from datasets import CellDataLoader
from finetuning_parameters.datasets import read_train_data
from src.augmentations import wider_train_transform
from training_loop import CellInstanceSegmentation

load_dotenv()

pl.seed_everything(0)


def objective(trial: Trial, data: pd.DataFrame, parameters: dict, cfg: EasyDict):
    lr = trial.suggest_float(name="lr", low=parameters['lr']['min'], high=parameters['lr']['max'])
    optimizer_name = trial.suggest_categorical(name='optimizer', choices=parameters['optimizer'])
    scheduler_name = trial.suggest_categorical(name='scheduler', choices=parameters['scheduler'])

    iou_scores = []
    folds = StratifiedKFold(n_splits=cfg.n_splits)
    for ii, (train_indices, val_indices) in enumerate(folds.split(data, data.cell_type)):
        train_df, val_df = data[train_indices], data[val_indices]

        dataloader = CellDataLoader(dataset_path=cfg.dataset_path,
                                    train_df=train_df, val_df=val_df,
                                    train_transform=wider_train_transform,
                                    batch_size=cfg.batch_size,
                                    num_workers=cfg.num_workers)

        train_dataloader, val_dataloader = dataloader.train_dataloader(), dataloader.val_dataloader()

        hparams = cfg.__dict__ | dict(lr=lr, optimizer_name=optimizer_name, scheduler_name=scheduler_name, kfold=ii)

        logger = WandbLogger(project="tuning_model_hparams", config=hparams)
        lr_monitor = LearningRateMonitor(logging_interval='step')

        model = CellInstanceSegmentation(cfg=EasyDict(hparams),
                                         train_dataloader=train_dataloader,
                                         val_dataloader=val_dataloader)

        trainer = pl.Trainer(logger=logger,
                             callbacks=[lr_monitor],
                             gpus=cfg.device,
                             fast_dev_run=True)

        trainer.fit(model, train_dataloader, val_dataloader)
        iou_scores.append(trainer.callback_metrics["val/iou_score"].item())

    return np.mean(iou_scores)


@click.command()
@click.option("--dataset-path", type=str, default=os.environ['dataset_path'])
@click.option("--device", type=str, required=True)
@click.option('--batch-size', type=int, default=4)
@click.option("--num-workers", type=int, default=10)
@click.option('--epochs', type=int, default=50)
def main(dataset_path: str, device: str, batch_size: int, num_workers: int, epochs: int):
    with open("params.yaml") as file:
        config = yaml.safe_load(file)

    n_splits = config['k_folds']
    parameters = config['parameters']
    data = read_train_data(Path(dataset_path))

    config = EasyDict(
        dataset_path=dataset_path,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        epochs=epochs,
        n_splits=n_splits,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, data, parameters, config), n_trials=1)


if __name__ == '__main__':
    main()
