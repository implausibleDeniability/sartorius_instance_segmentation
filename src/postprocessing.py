from typing import Union

import numpy as np
from torchvision.ops.boxes import nms
from pycocotools import mask as mutils
#from src.nms import mask_suppression


def remove_overlapping_pixels_two_masks(mutable_mask: np.ndarray, immutable_mask: np.ndarray):
    """Removes from mutable mask all pixels, that intersect with immutable_mask"""
    intersection = mutable_mask & immutable_mask
    mutable_mask -= intersection


def remove_overlapping_pixels(masks: np.ndarray) -> np.ndarray:
    """Removes all intersecting pixels in the sequence of masks
    
    Args: 
        masks: np.ndarray [N_MASKS, HEIGHT, WIDTH]
    Returns:
        np.ndarray [N_MASKS, HEIGHT, WIDTH]
    """
    masks = masks.copy()
    included_masks = np.zeros((masks.shape[1], masks.shape[2]))
    n_masks = len(masks)
    for i in range(n_masks):
        masks[i] -= np.logical_and(masks[i], included_masks)
        included_masks = np.logical_or(masks[i], included_masks)
    return masks


def postprocess_predictions(outputs: list[dict],
                            mask_threshold: float = 0.5,
                            score_threshold: Union[float, None] = None,
                            nms_threshold: Union[float, None] = None) -> list[dict]:
    """Postprocessing outputs from torchvision rcnn model

    Args:
        outputs - list of predictions, produced by torch Mask R-CNN
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
        indices = masks.any(axis=2).any(axis=1)
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


def postprocess_mmdet_predictions(
    mmdet_masks:list[tuple],
    nms_threshold: Union[float, None] = 0.1,
):
    """
    Postprocess predictions, produced by mmdetection Cascade R-CNN
    Args:
        mmdet_masks: data structure produced by mmdetection Cascade R-CNN, 
            list of tuples of tuples
        nms_threshold: threshold value for non-maximum suppression
    Returns:
        list of dicts, only dict's value is np.ndarray [N_MASKS, HEIGHT, WIDTH]
    """
    final_masks = []
    for image_prediction in mmdet_masks:
        bboxes_prediction, masks_prediction = image_prediction
        masks = []
        scores = []
        for class_id in range(len(masks_prediction)):
            masks.extend(map(mutils.decode, masks_prediction[class_id]))
            scores.extend(bboxes_prediction[class_id][:, 4])
        
        masks = np.array(masks)
        scores = np.array(scores)
        indices = masks.any(axis=2).any(axis=1)
        masks, scores = masks[indices], scores[indices]
        if nms_threshold:
            masks = mask_suppression(masks, scores, nms_threshold)
        non_overlapping_masks = remove_overlapping_pixels(masks)
        assert np.max(np.sum(non_overlapping_masks, axis=0)) <= 1, "Masks overlap"

        final_masks.append({
            'masks': non_overlapping_masks,
        })
    return final_masks