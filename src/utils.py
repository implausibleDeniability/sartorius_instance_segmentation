import os

import numpy as np
import random
import torch
from dotenv import load_dotenv


def annotation2mask(annotation: str) -> np.ndarray:
    """Converts annotation string of the image 520x704 to np.ndarray 520x704"""
    # -1 as RLE numerates from 1, not from 0
    segment_starts = np.array(annotation.split()[0::2], dtype=np.int32) - 1 
    segment_lengths = np.array(annotation.split()[1::2], dtype=np.int32)
    segment_ends = segment_starts + segment_lengths 
    
    load_dotenv()
    image_height = int(os.environ['IMAGE_HEIGHT'])
    image_width = int(os.environ['IMAGE_WIDTH'])
    flatten_mask = np.zeros(image_height * image_width)
    for i in range(len(segment_starts)):
        flatten_mask[segment_starts[i]:segment_ends[i]] = 1
    mask = flatten_mask.reshape([image_height, image_width])
    return mask


def get_box(mask: np.array) -> list:
    """ Get the bounding box of a given mask """
    y_coords, x_coords = np.where(mask)
    xmin = np.min(x_coords)
    xmax = np.max(x_coords)
    ymin = np.min(y_coords)
    ymax = np.max(y_coords)
    return [xmin, ymin, xmax, ymax]


def make_deterministic(seed: int):
    """Switches packages to deterministic behavior"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def images2device(images, device):
    images = list(image.to(device) for image in images)
    return images

def targets2device(targets, device):
    targets = [
        {key: value.to(device) for key, value in target.items()} for target in targets
    ]
    return targets