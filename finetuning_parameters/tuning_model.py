import os
import pickle
from pathlib import Path

import albumentations as A
import click
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import wandb
import yaml
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv
from easydict import EasyDict
from optuna import Trial
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold

from datasets import CellDataLoader
from finetuning_parameters.datasets import read_train_data
from training_loop import CellInstanceSegmentation

pl.seed_everything(0)
load_dotenv()

wider_train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.485,), std=(0.229,)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))


def objective(trial: Trial, data: pd.DataFrame, parameters: dict, cfg: EasyDict):
    lr = trial.suggest_float(name="lr", low=parameters['lr']['min'], high=parameters['lr']['max'])
    optimizer_name = trial.suggest_categorical(name='optimizer', choices=parameters['optimizer'])
    scheduler_name = trial.suggest_categorical(name='scheduler', choices=parameters['scheduler'])

    iou_scores = []
    folds = StratifiedKFold(n_splits=cfg.n_splits)
    for ii, (train_indices, val_indices) in enumerate(folds.split(data, data.cell_type)):
        train_df, val_df = data.iloc[train_indices], data.iloc[val_indices]

        dataloader = CellDataLoader(dataset_path=cfg.dataset_path,
                                    train_df=train_df, val_df=val_df,
                                    train_transform=wider_train_transform,
                                    batch_size=cfg.batch_size,
                                    num_workers=cfg.num_workers)

        train_dataloader, val_dataloader = dataloader.train_dataloader(), dataloader.val_dataloader()

        hparams = cfg.__dict__ | dict(lr=lr, optimizer_name=optimizer_name, scheduler_name=scheduler_name, kfold=ii)

        logger = WandbLogger(project="tuning_model_hparams",
                             config=hparams,
                             name=f"lr={lr},optim={optimizer_name},sched={scheduler_name},kfold={ii}")
        lr_monitor = LearningRateMonitor(logging_interval='step')

        model = CellInstanceSegmentation(cfg=EasyDict(hparams, steps_per_epochs=len(train_dataloader)),
                                         val_dataloader=val_dataloader)

        trainer = pl.Trainer(logger=logger,
                             max_epochs=cfg.epochs,
                             callbacks=[lr_monitor],
                             gpus=cfg.device,
                             fast_dev_run=True)

        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, val_dataloader)
        iou_scores.append(trainer.callback_metrics["test/iou_score"].item())

    return np.mean(iou_scores)


@click.command()
@click.option("--dataset-path", type=str, default=os.environ.get('dataset_path'))
@click.option("--device", type=str, required=True, help="cpu, cuda:0, ...")
@click.option('--batch-size', type=int, default=4)
@click.option("--num-workers", type=int, default=10)
@click.option('--epochs', type=int, default=50)
@click.option("--token", type=str, default="", help="WANDB_API_KEY")
def main(dataset_path: str, device: str, batch_size: int, num_workers: int, epochs: int, token: str):
    device = 0 if device == "cpu" else [int(device[-1])]

    if token:
        os.environ['WANDB_API_KEY'] = token

    with open("params.yaml") as file:
        config = yaml.safe_load(file)

    n_splits = config['k_folds']
    parameters = config['parameters']
    data = read_train_data(Path(dataset_path))

    config = EasyDict(
        dataset_path=Path(dataset_path),
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        epochs=epochs,
        n_splits=n_splits,
        mask_threshold=0.5,
        score_threshold=0.05,
        nms_threshold=None,
    )

    study = optuna.create_study(direction="maximize", storage='sqlite:///tuning_parameters.db', load_if_exists=True)
    study.optimize(lambda trial: objective(trial, data, parameters, config), n_trials=1)

    saving_dir = Path(".") / "tuning_parameters"
    saving_dir.mkdir()
    with open(saving_dir / "optuna_logs.pickle", "wb") as file:
        pickle.dump(study, file)

    study.trials_dataframe().to_csv(saving_dir / "exp_data.csv", index=False)

    fig = plot_optimization_history(study)
    fig.write_image(saving_dir / "opt_history.jpg", scale=2)
    wandb.save(saving_dir / "opt_history.jpg")

    fig = plot_contour(study)
    fig.write_image(saving_dir / "contour.jpg", scale=2)
    wandb.save(saving_dir / "contour.jpg")

    fig = plot_param_importances(study)
    fig.write_image(saving_dir / "param_importance.jpg", scale=2)
    wandb.save(saving_dir / "param_importance.jpg")


if __name__ == '__main__':
    main()
