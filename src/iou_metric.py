import numpy as np

eps = 1e-6


def iou(true_mask: np.array, pred_mask: np.array):
    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum() + eps

    return intersection.sum() / union.sum()


def precision(true_masks: np.array, pred_masks: np.array, iou_threshold: float):
    true_positive = 0
    for mask1 in true_masks:
        for mask2 in pred_masks:
            if iou(mask1, mask2) > iou_threshold:
                true_positive += 1

    metric = true_positive / (len(true_masks) + len(pred_masks) - true_positive)
    return metric


def iou_map(true_masks: np.array, pred_masks: np.array):
    average_precision = []
    for threshold in np.arange(start=0.5, stop=1, step=0.05):
        average_precision.append(precision(true_masks, pred_masks, threshold))

    return np.asarray(average_precision).mean()
