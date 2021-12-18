import os
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from easydict import EasyDict
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from src.augmentations import train_transform
from src.dataset import COCODataset
from src.utils import make_deterministic, images2device, targets2device


@click.command(name="Pretraining model on another dataset")
@click.option("--device", type=str, required=True)
@click.option("--experiment-name", type=str, required=True)
def main(device: str, experiment_name: str):
    assert device.startswith("cuda:") or device == "cpu"
    load_dotenv()
    make_deterministic(seed=42)

    global config
    config = EasyDict(
        dataset_path=Path(os.environ["dataset_path"]),
        weights_path=Path(os.environ["weights_path"]),
        device=device,
        val_size=0.2,
        batch_size=2,
        num_workers=10,
        epochs=20
    )
    pretrain_dir = config.dataset_path / "LIVECell_dataset_2021"
    config.images = pretrain_dir / "images"
    config.train_annotations = pretrain_dir / "annotations" / "LIVECell" / "livecell_coco_train.json"

    # configuration
    wandb.init(
        project="sartorius_instance_segmentation",
        # entity="implausible_denyability",
        name=experiment_name,
        config=config,
    )

    # DataLoaders
    train_dataset = COCODataset(image_dir=config.images,
                                annotation_file=config.train_annotations,
                                transform=train_transform)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Training
    model = models.detection.maskrcnn_resnet50_fpn(
        num_classes=2, progress=False, box_detections_per_img=400
    )
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        epochs=config.epochs,
        steps_per_epoch=len(train_dataloader),
        max_lr=1e-3,
    )
    training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config.epochs,
        train_loader=train_dataloader,
    )

    # Save weights
    weights_dir = config.weights_path
    weights_dir.mkdir(exist_ok=True)
    weights_path = (
            weights_dir / f"{experiment_name}-{datetime.now().__str__()[:-7]}.ckpt"
    )
    torch.save(model.state_dict(), weights_path)
    wandb.save(str(weights_path.absolute()))


def training(model, optimizer, scheduler, epochs, train_loader):
    for epoch in range(epochs):
        loss, mask_loss = train_epoch(model, train_loader, optimizer, scheduler)

        wandb.log(
            {
                "train_loss": loss,
                "train_mask_loss": mask_loss,
                "lr": scheduler.get_last_lr()[0],
            }
        )


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    losses = []
    mask_losses = []
    for images, targets in tqdm(loader):
        loss, mask_loss = train_batch(model, images, targets, optimizer)
        losses.append(loss)
        mask_losses.append(mask_loss)
        scheduler.step()
    return np.mean(losses), np.mean(mask_losses)


def train_batch(model, images, targets, optimizer):
    optimizer.zero_grad()
    images = images2device(images, config.device)
    targets = targets2device(targets, config.device)
    output = model(images, targets)
    loss = sum(single_loss for single_loss in output.values())
    loss.backward()
    optimizer.step()
    return loss.item(), output["loss_mask"].item()


if __name__ == "__main__":
    main()
