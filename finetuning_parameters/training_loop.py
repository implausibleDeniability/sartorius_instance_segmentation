import numpy as np
import pytorch_lightning as pl
import torch
from easydict import EasyDict
from torchvision.models.detection import maskrcnn_resnet50_fpn

from finetuning_parameters.select_parameters import get_optimizer, get_scheduler
from src.iou_metric import iou_map
from src.postprocessing import postprocess_predictions


class CellInstanceSegmentation(pl.LightningModule):
    def __init__(self, cfg: EasyDict, train_dataloader, val_dataloader):
        super(CellInstanceSegmentation, self).__init__()

        self.model = maskrcnn_resnet50_fpn(num_classes=4,
                                           progress=False,
                                           box_detections_per_img=500,
                                           trainable_backbone_layers=5)
        self.cfg = cfg

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def _shared_step(self, batch):
        self.model.train()

        images, targets = batch
        outputs = self.model.forward(images, targets)
        loss_step = sum(loss for loss in outputs.values())
        loss_mask = outputs['loss_mask'].item()

        return {"loss": loss_step, "loss_mask": loss_mask.item()}

    def training_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)

        self.log("train/loss_step", outputs['loss'].item())
        self.log("train/loss_mask_step", outputs['loss_mask'])

        return outputs

    def training_epoch_end(self, outputs) -> None:
        loss_epoch = torch.hstack([loss['loss_step'] for loss in outputs]).mean()
        loss_mask_epoch = torch.hstack([loss['loss_mask'] for loss in outputs]).mean()

        self.log("train/loss_epoch", loss_epoch.item())
        self.log("train/loss_mask_epoch", loss_mask_epoch.item())

        if self.current_epoch % 7 == 0 or self.current_epoch + 5 > self.cfg.epochs:
            iou_score = self.calculate_iou(dataloader=self.train_dataloader)
            self.log("train/iou_score", torch.as_tensor(iou_score).item())

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        return outputs

    def validation_epoch_end(self, outputs) -> None:
        loss_epoch = torch.hstack([loss['loss_step'] for loss in outputs]).mean()
        loss_mask_epoch = torch.hstack([loss['loss_mask'] for loss in outputs]).mean()

        self.log("val/loss_epoch", loss_epoch.item())
        self.log("val/loss_mask_epoch", loss_mask_epoch.item())

        if self.current_epoch % 7 == 0 or self.current_epoch + 5 > self.cfg.epochs:
            iou_score = self.calculate_iou(dataloader=self.val_dataloader)
            self.log("val/iou_score", torch.as_tensor(iou_score).item())

    def configure_optimizers(self):
        optimizer = get_optimizer(name=self.cfg.optimizer_name, model=self.model, lr=self.cfg.lr)

        if self.cfg.scheduler_name == 'None':
            return optimizer
        else:
            scheduler = get_scheduler(name=self.cfg.scheduler_name,
                                      optimizer=optimizer,
                                      steps_per_epochs=self.cfg.steps_per_epochs,
                                      epochs=self.cfg.epochs,
                                      lr=self.cfg.lr)

            return [optimizer], [scheduler]

    def calculate_iou(self, dataloader):
        self.model.eval()

        scores = []
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx > 31:
                break

            images, targets = batch

            predictions = self.model(images)
            processed_outputs = postprocess_predictions(
                predictions,
                self.cfg.mask_threshold,
                self.cfg.score_threshold,
                self.cfg.nms_threshold
            )

            for ii in range(len(images)):
                predicted_masks = processed_outputs[ii]["masks"]
                true_masks = targets[ii]["masks"]
                iou = iou_map(predicted_masks, true_masks.numpy())
                scores.append(iou)

        return np.mean(scores)
