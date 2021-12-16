import json
from collections import defaultdict
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict
from rasterio.features import rasterize
from shapely.geometry import Polygon
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
            transformed = self.transform(image=image, masks=masks, bboxes=boxes, category_ids=labels)

            image = transformed['image']
            masks = transformed['masks']
            boxes = transformed['bboxes']

        boxes = np.asarray(boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Training on all samples (setting iscrown=False)
        iscrowd = torch.zeros((len(masks),), dtype=torch.int64)

        target = {
            'masks': torch.as_tensor(np.array(masks)),
            'labels': torch.as_tensor(np.array(labels), dtype=torch.int64),
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'iscrowd': iscrowd,
            'area': torch.as_tensor(area),
            # 'image_id': image_id,  # For logging purposes
        }

        return image, target

    def __len__(self):
        return len(self.data_csv)


def preprocess_annotations(data: dict) -> (dict, dict, dict):
    """Returns dictionaries that have as a key ids of images that maps to segmentations, boxes and areas
    Args:
        data - coco json dict

    Returns:
        dict[image_id] -> list of contours' masks
        dict[image_id] -> list of areas of each box
        dict[image_id] -> list of outer boxes
    """
    image_to_segmentation = defaultdict(list)
    image_to_boxes = defaultdict(list)
    image_to_area = defaultdict(list)
    for segm_id, metadata in data['annotations'].items():
        image_id = metadata['image_id']
        image_to_segmentation[image_id].append(metadata['segmentation'][0])
        image_to_area[image_id].append(metadata['area'])
        image_to_boxes[image_id].append(metadata['bbox'])

    return image_to_segmentation, image_to_area, image_to_boxes


def contours_to_masks(contours: list):
    """Converts COCO format of masks (polygons) to binary masks"""
    masks = []
    for segmentation in contours:
        segmentation = np.asarray(segmentation).reshape(-1, 2)
        polygon = Polygon(segmentation)
        mask = rasterize([polygon], out_shape=(520, 704))
        masks.append(mask)

    masks = np.asarray(masks)
    return masks


def xywh2xyxy(boxes):
    """Converts COCO format boxes (x_min, y_min, height, width) for passing to model (x_min, y_min, x_max, y_max)"""

    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    boxes[:, 2] += xmin
    boxes[:, 3] += ymin

    return boxes


class COCODataset(Dataset):
    """Dataset for reading COCO annotations format"""

    def __init__(self, image_dir: Path, annotation_file: Path, transform: A.Compose = None):
        super(COCODataset, self).__init__()

        self.transform = transform

        assert annotation_file.is_file()
        assert image_dir.is_dir()

        self.image_paths = {image.name: image.absolute() for image in image_dir.glob("**/*.*")}

        with open(annotation_file) as file:
            self.annotations = json.load(file)

        self.segmentations, self.areas, self.boxes = preprocess_annotations(self.annotations)

        self.image_id_to_image = [(image['id'], image['file_name']) for image in self.annotations['images']]

    def _load_image(self, name: str):
        image_path = self.image_paths[name]
        image = io.imread(str(image_path))
        return image

    def _load_target(self, image_id: int):
        segmentations = self.segmentations[image_id]
        area = self.areas[image_id]
        boxes = self.boxes[image_id]

        masks = contours_to_masks(segmentations)
        boxes = np.array(boxes)
        boxes = xywh2xyxy(boxes)
        area = np.array(area)

        return masks, boxes, area

    def __getitem__(self, item):
        image_id, image_name = self.image_id_to_image[item]
        image = self._load_image(image_name)

        masks, boxes, area = self._load_target(image_id)

        if self.transform:
            categories = np.ones(shape=len(masks))
            transformed = self.transform(image=image, masks=list(masks), bboxes=list(boxes), category_ids=categories)

            image = transformed['image']
            masks = transformed['masks']
            boxes = transformed['bboxes']

        target = {
            'masks': torch.as_tensor(np.asarray(masks)),
            'boxes': torch.as_tensor(np.asarray(boxes), dtype=torch.float32),
            'labels': torch.as_tensor([1] * len(masks), dtype=torch.int64),
            'iscrowd': torch.as_tensor([1] * len(masks), dtype=torch.int64),
            'ares': torch.as_tensor(np.asarray(area)),
        }

        return image, target

    def __len__(self):
        return len(self.image_id_to_image)
