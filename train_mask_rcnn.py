import os
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from easydict import EasyDict
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from src.augmentations import train_transform, wider_train_transform
from src.dataset import CellDataset
from src.image_logger import ImageLogger
from src.iou_metric import iou_map
from src.postprocessing import postprocess_predictions
from src.utils import make_deterministic, images2device, targets2device


@click.command()
@click.option("--device", required=True, type=str, help="Device to train on, e.x: cuda:0 or cpu")
@click.option("--exp_name", required=True, type=str, help="Name of experiment")
def main(device: str, exp_name: str):
    load_dotenv()
    make_deterministic(seed=42)

    # TODO: folder 'experiments' that contain all possible configurations
    global config
    config = EasyDict(
        dataset_path=Path(os.environ["dataset_path"]),
        weights_path=Path(os.environ["weights_path"]),
        device=device,
        val_size=0.2,
        batch_size=4,
        num_workers=25,
        epochs=50,
        mask_threshold=0.42,
        score_threshold=0.49,
        nms_threshold=0.32,
    )

    # configuration
    experiment_name = exp_name
    wandb.init(
        project="sartorius_instance_segmentation",
        entity="implausible_denyability",
        name=experiment_name,
        config=config,
    )

    # DataLoaders
    train_dataset = CellDataset(cfg=config, mode="train", transform=wider_train_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    val_dataset = CellDataset(cfg=config, mode="val", transform=train_transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # Training
    model = models.detection.maskrcnn_resnet50_fpn(
        num_classes=4, progress=False, box_detections_per_img=500
    )
    wandb.watch(model, log_freq=100)
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters())
    total_num_iterations = len(train_dataloader) * config.max_epochs
    scheduler = CyclicLR(optimizer=optimizer,
                         base_lr=1e-5,
                         max_lr=1e-3,
                         mode="triangular2",
                         step_size_up=total_num_iterations / 7)

    image_logger = ImageLogger(train_dataset=train_dataset, val_dataset=val_dataset, n_images=10, device=config.device)
    training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config.epochs,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        image_logger=image_logger,
    )

    # Save weights
    weights_dir = config.weights_path
    weights_dir.mkdir(exist_ok=True)
    weights_path = weights_dir / f"{experiment_name}-{datetime.now().__str__()[:-7]}.ckpt"
    torch.save(model.state_dict(), weights_path)
    wandb.save(str(weights_path.absolute()))


def training(model, optimizer, scheduler, epochs, train_loader, val_loader, image_logger):
    for epoch in range(epochs):
        loss, mask_loss = train_epoch(model, train_loader, optimizer, scheduler)
        if epoch % 7 == 0 or epoch + 5 > config.epochs:
            image_logger.log_images(model)
            train_iou = eval_epoch(model, train_loader)
            val_iou = eval_epoch(model, val_loader)
        wandb.log(
            {
                "train_loss": loss,
                "train_mask_loss": mask_loss,
                "train_iou": train_iou,
                "val_iou": val_iou,
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


def eval_epoch(model, loader):
    model.eval()
    ious = []
    for batch_idx, (images, targets) in enumerate(tqdm(loader)):
        if batch_idx > 31:
            break
        images = images2device(images, config.device)
        outputs = model(images)
        processed_outputs = postprocess_predictions(
            outputs, config.mask_threshold, config.score_threshold, config.nms_threshold
        )
        for i in range(len(images)):
            predicted_masks = processed_outputs[i]["masks"]
            true_masks = targets[i]["masks"]
            iou = iou_map(predicted_masks, true_masks.numpy())
            ious.append(iou)
    return np.mean(ious)


if __name__ == "__main__":
    main()
