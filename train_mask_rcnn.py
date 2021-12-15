import os
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from easydict import EasyDict
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

import wandb
from src.augmentations import train_transform
from src.dataset import CellDataset
from src.utils import make_deterministic, images2device, targets2device


def main():
    load_dotenv()
    make_deterministic(seed=42)
    current_dir = Path(".")  # In my case, it is sartorius_instance_segmentation
    # TODO: folder 'experiments' that contain all possible configurations
    config = EasyDict(
        dataset_path=Path(os.environ["dataset_path"]),
        device="cuda:1",
        val_size=0.2,
        batch_size=6,
        num_workers=30,
        epochs=1,
        mask_threshold=0.5,
        score_threshold=0.2,
        nms_threshold=None,
    )

    # configuration
    experiment_name = "temp"
    wandb.init(
        project="sartorius_instance_segmentation",
        entity="implausible_denyability",
        name=experiment_name,
        config=config,
    )

    global device
    device = config.device

    # DataLoaders
    train_dataloader = DataLoader(
        dataset=CellDataset(cfg=config, mode="train", transform=train_transform),
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # Training
    model = models.detection.maskrcnn_resnet50_fpn(num_classes=2, progress=False, box_detections_per_img=500)
    wandb.watch(model, log_freq=100)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        epochs=config.epochs,
        steps_per_epoch=len(train_dataloader),
        max_lr=1e-3,
    )
    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config.epochs,
        loader=train_dataloader,
    )

    # Save weights
    weights_dir = current_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    weights_path = weights_dir / f"maskrcnn-{experiment_name}-{datetime.now().__str__()[:-7]}.ckpt"
    torch.save(model.state_dict(), weights_path)
    wandb.save(str(weights_path.absolute()))


def train_batch(model, images, targets, optimizer):
    optimizer.zero_grad()
    images, targets = images2device(images, device), targets2device(targets, device)
    output = model(images, targets)
    loss = sum(single_loss for single_loss in output.values())
    loss.backward()
    optimizer.step()
    return loss.item(), output["loss_mask"].item()


def train(model, optimizer, scheduler, epochs, loader):
    for epoch in range(epochs):
        model.train()
        for images, targets in tqdm(loader):
            loss, mask_loss = train_batch(model, images, targets, optimizer)
            wandb.log(
                {
                    "loss/train": loss,
                    "mask_loss/train": mask_loss,
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            scheduler.step()


if __name__ == "__main__":
    main()
