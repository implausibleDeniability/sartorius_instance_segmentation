from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from skimage import io
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from src.utils import annotation2mask, get_box

default_transform = A.Compose([
    A.Normalize(mean=(0.485,), std=(0.229,)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))


def read_train_data(dataset_path: Path) -> pd.DataFrame:
    """Reading annotations for labeled images"""
    data_csv = pd.read_csv(dataset_path / "train.csv")
    data_csv = data_csv[['id', 'annotation', 'cell_type']]
    data_csv = data_csv.groupby(['id', 'cell_type'])['annotation'].agg(lambda x: list(x)).reset_index()
    data_csv.cell_type = LabelEncoder().fit_transform(data_csv.cell_type) + 1

    return data_csv[['id', 'cell_type', 'annotation']]


class CellDataset(Dataset):
    def __init__(self, dataset_path: Path, df: pd.DataFrame, transform: A.Compose = None):
        """
        Args:
            dataset_path - Path to dataset folder (needed for reading images)
            df - pd.DataFrame, whole data or k-split
            transform - albumentations transform for images
        """

        self.transform = transform
        self.image_folder = dataset_path / "train"
        self.data_csv = df

    def __getitem__(self, item):
        image_id, cell_type, annotations = self.data_csv.iloc[item]
        image_path = self.image_folder / (image_id + ".png")
        image = io.imread(str(image_path))

        masks = list(map(lambda line: annotation2mask(line), annotations))
        boxes = list(map(lambda mask: get_box(mask), masks))
        labels = list(map(lambda _: cell_type, masks))

        if self.transform:
            transformed = self.transform(image=image, masks=masks, bboxes=boxes, category_ids=labels)
            image = transformed['image']
            masks = transformed['masks']
            boxes = transformed['bboxes']
            labels = transformed['category_ids']
            boxes = np.asarray(boxes)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            raise Exception("Please use transform")

        iscrowd = torch.zeros((len(masks),), dtype=torch.int64)

        target = {
            'masks': torch.stack(masks),
            'labels': torch.as_tensor(np.array(labels), dtype=torch.int64),
            'boxes': torch.as_tensor(boxes),
            'iscrowd': iscrowd,
            'area': torch.as_tensor(area),
        }

        return image, target

    def __len__(self):
        return len(self.data_csv)


class CellDataLoader(pl.LightningDataModule):
    def __init__(self,
                 dataset_path: Path,
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 train_transform: A.Compose,
                 batch_size: int,
                 num_workers: int):
        super(CellDataLoader, self).__init__()

        self.train_dataset = CellDataset(dataset_path=dataset_path, df=train_df, transform=train_transform)
        self.val_dataset = CellDataset(dataset_path=dataset_path, df=val_df, transform=default_transform)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          collate_fn=lambda x: tuple(zip(*x)))

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=lambda x: tuple(zip(*x)))
