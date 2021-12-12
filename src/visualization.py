from functools import reduce

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def tensor_to_image(im: torch.Tensor):
    im = np.transpose(im.numpy(), axes=(1, 2, 0))
    # noinspection PyArgumentList
    im = (im - im.min()) / (im.max() - im.min())
    im = np.asarray(im * 255., dtype=np.uint8)
    return im


def put_mask_on_image(image: torch.Tensor, masks: np.ndarray) -> np.ndarray:
    image = tensor_to_image(image)
    image_mask = reduce(lambda x, y: x + y, masks)
    image_mask[image_mask > 1] = 1
    yellow_mask = np.stack([image_mask, image_mask, np.zeros_like(image_mask)], axis=2)
    image_with_mask = np.array(image + 50 * yellow_mask, dtype=np.uint8)
    image_with_mask[image_with_mask > 256] = 255
    return image_with_mask


def plot_two_masks(image: torch.Tensor, predicted_mask: np.ndarray, true_mask: np.ndarray,
                   color=(255, 255, 0), filename=None):
    image_mask1 = put_mask_on_image(image, predicted_mask)
    image_mask2 = put_mask_on_image(image, true_mask)
    plt.figure(figsize=(20, 8))
    plt.subplot(121)
    plt.imshow(image_mask1)
    plt.subplot(122)
    plt.imshow(image_mask2)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=80)


def plot_mask_bbox(image: torch.Tensor, boxes: np.array, masks: np.array, figure_scale: int = 15):
    """Visualization instance of dataset after augmentation

    Args:
        image: torch.Tensor with shape [1, height, width]
        boxes: torch.Tensor with shape [N, 4] with N being number of boxes
        masks: torch.Tensor with shape [N, height, width]
    """
    image_with_mask = put_mask_on_image(image, masks)

    # Drawing red rectangle for each instances
    red_color = (255, 0, 0)
    for x1, y1, x2, y2 in boxes:
        image_with_mask = cv2.rectangle(image_with_mask.copy(), pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)),
                                        color=red_color, thickness=2)

    plt.figure(figsize=(figure_scale, figure_scale))
    plt.imshow(image_with_mask)
    plt.xticks([])
    plt.yticks([])
    plt.show()
