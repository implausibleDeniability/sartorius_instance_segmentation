from typing import Union

import numpy as np
import torch
from torchvision.ops.boxes import nms


def remove_overlapping_pixels_two_masks(mutable_mask: np.ndarray, immutable_mask: np.ndarray):
    """Removes from mutable mask all pixels, that intersect with immutable_mask"""
    intersection = mutable_mask & immutable_mask
    mutable_mask -= intersection


def remove_overlapping_pixels(masks: np.ndarray):
    """Removes all intersecting pixels in the sequence of masks
    
    Args: 
        np.ndarray [N_MASKS, HEIGHT, WIDTH]
    Returns:
        np.ndarray [N_MASKS, HEIGHT, WIDTH]
    """
    masks = masks.copy()
    n_masks = len(masks)
    for i in range(n_masks):
        for j in range(i + 1, n_masks):
            remove_overlapping_pixels_two_masks(masks[i], masks[j])
    return masks


def postprocess_predictions(outputs: list[dict],
                            mask_threshold: float = 0.5,
                            score_threshold: Union[float, None] = None,
                            nms_threshold: Union[float, None] = None):
    """Postprocessing outputs from torchvision rcnn model

    Args:
        outputs - list of predictions
        mask_threshold - value for cutting object from background
        score_threshold - value for removing objects that model is not confident
        nms_threshold - iou threshold for removing overlapping instances

    Returns:
        List of dict(mask, boxes, scores)
    """

    result = []
    for output in outputs:
        
        scores = output['scores'].detach().cpu()
        masks = output['masks'].detach().cpu().squeeze()
        boxes = output['boxes'].detach().cpu()
        
        masks = (masks >= mask_threshold).int()

        # Now some masks can be empty (all zeros), we need to exclude them
        indices = torch.as_tensor([torch.sum(mask) > 0 for mask in masks])
        masks, boxes, scores = masks[indices], boxes[indices], scores[indices]

        if score_threshold:
            indices = scores >= score_threshold
            masks, boxes, scores = masks[indices], boxes[indices], scores[indices]

        if nms_threshold:
            indices = nms(boxes=boxes, scores=scores, iou_threshold=nms_threshold)
            masks, boxes, scores = masks[indices], boxes[indices], scores[indices]
        
        non_overlapping_masks = remove_overlapping_pixels(masks.numpy())
        assert np.max(np.sum(non_overlapping_masks, axis=0)) <= 1, "Masks overlap"

        result.append({
            'masks': non_overlapping_masks,
            'boxes': boxes,
            'scores': scores
        })

    return result
