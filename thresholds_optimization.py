import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from dotenv import load_dotenv
from easydict import EasyDict
from tqdm import tqdm
import pickle
import optuna

from src.postprocessing import postprocess_predictions
from src.dataset import CellDataset
from src.augmentations import train_transform
from src.utils import images2device
from src.iou_metric import iou_map

def main():
    load_dotenv()

    experiment_name = 'bsln500ep'
    weights_name = f'{experiment_name}.ckpt'
    weights_path = Path(os.environ["weights_path"]) / weights_name

    global config
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

    model = maskrcnn_resnet50_fpn(progress=False, num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(config.device)))
    model.to(config.device)
    model.eval()

    train_dataloader = DataLoader(
        dataset=CellDataset(cfg=config, mode="val", transform=train_transform),
        num_workers=30,
        batch_size=config.batch_size,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    os.makedirs('thresholds', exist_ok=True)
    study = optuna.create_study(direction='maximize')
    for i in range(40):
        study.optimize(lambda trial: evaluate_thresholds(trial, model, train_dataloader), n_trials=3)
        with open(f"thresholds/{experiment_name}.pickle", 'wb') as file:
            pickle.dump(study, file)
        study.trials_dataframe().to_csv(f"thresholds/{experiment_name}.csv", index=False)


def evaluate_thresholds(trial, model, dataloader):
    mask_thr = trial.suggest_float('mask_thr', 0.0, 0.99)
    score_thr = trial.suggest_float('score_thr', 0.0, 0.99)
    nms_thr = trial.suggest_float('nms_thr', 0.0, 1.0)
    ious = []
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader)):
        images = images2device(images, config.device)
        outputs = model(images)
        processed_outputs = postprocess_predictions(outputs, mask_thr, score_thr, nms_thr)
        for i in range(len(images)):
            predicted_masks = processed_outputs[i]['masks']
            true_masks = targets[i]['masks']
            # print(predicted_masks.shape, true_masks.shape)
            iou = iou_map(predicted_masks, true_masks.numpy())
            ious.append(iou)
    return np.mean(ious)

if __name__ == "__main__":
    main()
