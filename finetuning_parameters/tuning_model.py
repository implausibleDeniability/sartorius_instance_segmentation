import os
import pickle
from pathlib import Path

import albumentations as A
import click
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
from sklearn.model_selection import train_test_split

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


def objective(trial: Trial, data: pd.DataFrame, parameters: dict, cfg: EasyDict) -> float:
    """Choosing hparam function that makes the experiment
    Args:
        trial: object that suggests hparam
        data: dataframe with annotations
        parameters: dict, containing names of optimizers and schedulers
        cfg: non-tuning parameters for training model (batch size, ...)

    Returns:
        Validation iou score of the experiment
    """
    lr = trial.suggest_float(name="lr", low=parameters['lr']['min'], high=parameters['lr']['max'])
    optimizer_name = trial.suggest_categorical(name='optimizer', choices=parameters['optimizer'])
    scheduler_name = trial.suggest_categorical(name='scheduler', choices=parameters['scheduler'])

    train_df, val_df = train_test_split(data, test_size=0.2, stratify=data.cell_type)

    dataloader = CellDataLoader(dataset_path=cfg.dataset_path,
                                train_df=data, val_df=val_df,
                                train_transform=wider_train_transform,
                                batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers)

    train_dataloader, val_dataloader = dataloader.train_dataloader(), dataloader.val_dataloader()

    hparams = cfg.__dict__ | dict(lr=lr, optimizer_name=optimizer_name, scheduler_name=scheduler_name)

    logger = WandbLogger(project="sartorius_instance_segmentation",
                         config=hparams,
                         name=f"tune?lr={lr},optim={optimizer_name},sched={scheduler_name}",
                         save_dir=Path("logs").__str__())
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model = CellInstanceSegmentation(cfg=EasyDict(hparams, steps_per_epochs=len(train_dataloader)),
                                     val_dataloader=val_dataloader)

    trainer = pl.Trainer(logger=logger,
                         max_epochs=cfg.epochs,
                         callbacks=[lr_monitor],
                         gpus=cfg.device)

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, val_dataloader)
    iou_score = trainer.callback_metrics["test/iou_score"].item()
    print(f"test/iou_score: {iou_score}")

    wandb.finish()

    return iou_score


@click.command()
@click.option("--dataset-path", type=str, default=os.environ.get('dataset_path'))
@click.option("--device", type=str, required=True, help="cpu, cuda:0, ...")
@click.option('--batch-size', type=int, default=2)
@click.option("--num-workers", type=int, default=4)
@click.option('--epochs', type=int, default=10)
@click.option("--token", type=str, default="", help="WANDB_API_KEY")
def main(dataset_path: str, device: str, batch_size: int, num_workers: int, epochs: int, token: str):
    saving_dir = Path("logs")
    saving_dir.mkdir(exist_ok=True)

    device = 0 if device == "cpu" else [int(device[-1])]

    if token:
        os.environ['WANDB_API_KEY'] = token

    with open("finetuning_parameters/params.yaml") as file:
        config = yaml.safe_load(file)

    parameters = config['parameters']
    data = read_train_data(Path(dataset_path))

    config = EasyDict(
        dataset_path=Path(dataset_path),
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        epochs=epochs,
        mask_threshold=0.5,
        score_threshold=0.05,
        nms_threshold=None,
    )

    study = optuna.create_study(direction="maximize", storage='sqlite:///logs/tuning_parameters.db',
                                load_if_exists=True)
    study.optimize(lambda trial: objective(trial, data, parameters, config), n_trials=20)

    # Dumping experiment logs and visualisations
    with open(saving_dir / "optuna_logs.pickle", "wb") as file:
        pickle.dump(study, file)

    study.trials_dataframe().to_csv(saving_dir / "exp_data.csv", index=False)

    fig = plot_optimization_history(study)
    fig.write_image(saving_dir / "opt_history.jpg", scale=2)
    wandb.save(str(saving_dir.absolute() / "opt_history.jpg"))

    fig = plot_contour(study)
    fig.write_image(saving_dir / "contour.jpg", scale=2)
    wandb.save(str(saving_dir.absolute() / "contour.jpg"))

    fig = plot_param_importances(study)
    fig.write_image(saving_dir / "param_importance.jpg", scale=2)
    wandb.save(str(saving_dir.absolute() / "param_importance.jpg"))


if __name__ == '__main__':
    main()
