import albumentations as A
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict
from skimage import io
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.utils import annotation2mask, get_box


class CellDataset(Dataset):
    def __init__(self, cfg: EasyDict, mode: str, transform: A.Compose = None):
        assert mode == 'train' or mode == 'val'

        self.transform = transform
        self.image_folder = cfg.dataset_path / "train"

        data_csv = pd.read_csv(cfg.dataset_path / "train.csv")
        data_csv = data_csv[['id', 'annotation', 'cell_type']]
        data_csv = data_csv.groupby(['id', 'cell_type'])['annotation'].agg(lambda x: list(x)).reset_index()

        # Now data_csv has 606 rows with 3 columns:
        # - id - id of image,
        # - cell_type - type of cell,
        # - annotation - list of rle strings
        if mode == 'train':
            self.data_csv, _ = train_test_split(data_csv, test_size=cfg.val_size, random_state=0)
        else:
            _, self.data_csv = train_test_split(data_csv, test_size=cfg.val_size, random_state=0)

    def __getitem__(self, item):
        # TODO: Test how much time does .iloc take
        # if I (maxim) am not mistaken, that can be quite long
        image_id, _, annotations = self.data_csv.iloc[item]
        image_path = self.image_folder / (image_id + ".png")
        image = io.imread(str(image_path))

        # TODO: consider using rle_decode instead of annotation2mask
        masks = list(map(lambda line: annotation2mask(line), annotations))
        boxes = list(map(lambda mask: get_box(mask), masks))
        labels = list(map(lambda _: 1, masks))

        if self.transform:
            transformed = self.transform(image=image, masks=masks, bboxes=boxes, category_ids= labels)

            image = transformed['image']
            masks = transformed['masks']
            boxes = transformed['bboxes']

        boxes = np.asarray(boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Training on all samples (setting iscrown=False)
        iscrowd = torch.zeros((len(masks),), dtype=torch.int64)

        target = {
            'masks': torch.as_tensor(masks),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'iscrowd': iscrowd,
            'area': torch.as_tensor(area),
            # 'image_id': image_id,  # For logging purposes
        }


        return image, target

    def __len__(self):
        return len(self.data_csv)