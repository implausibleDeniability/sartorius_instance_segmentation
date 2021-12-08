from functools import reduce

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def tensor_to_image(im: torch.Tensor):
    im = np.transpose(im.numpy(), axes=(1, 2, 0))
    im = (im - im.min()) / (im.max() - im.min())
    im = np.asarray(im * 255., dtype=np.uint8)

    return im


def show_image(im: torch.Tensor, boxes, masks):
    """
    Args:
        im: torch.Tensor with shape [1, height, width]
        boxes: torch.Tensor with shape [N, 4] with N being number of boxes
        masks: torch.Tensor with shape [N, height, width]
        
    Visualization instance of dataset after augmentation
    """
    # torch image preprocessed for plotting with matplotlib
    im = tensor_to_image(im)

    image_mask = reduce(lambda x, y: x + y, masks)

    image_mask[image_mask > 1] = 1
    yellow_mask = np.stack([image_mask, image_mask, np.zeros_like(image_mask)], axis=2)

    image_with_mask = np.array(im + 50 * yellow_mask, dtype=np.uint8)

    # Drawing red rectangle for each instances
    red_color = (255, 0, 0)
    for x1, y1, x2, y2 in boxes:
        image_with_mask = cv2.rectangle(image_with_mask.copy(), pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=red_color, thickness=2)

    plt.figure(figsize=(15, 15))
    plt.imshow(image_with_mask)
    plt.xticks([])
    plt.yticks([])
    plt.show()
