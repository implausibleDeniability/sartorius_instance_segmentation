import numpy as np

eps = 1e-6


def fast_iou(predicted_masks: np.ndarray, true_masks: np.ndarray) -> np.float64:
    """
    Args:
        predicted_masks: np.array [N_MASKS, HEIGHT, WIDTH]
        true_masks: np.array [M_MASKS, HEIGHT, WIDTH]
    """
    n_predicted = predicted_masks.shape[0]
    n_true = true_masks.shape[0]
    height = true_masks.shape[1]
    width = true_masks.shape[2]
    
    predicted_masks = predicted_masks.reshape(1, n_predicted, height, width)
    true_masks = true_masks.reshape(n_true, 1, height, width)
    logical_or = np.logical_or(predicted_masks, true_masks).sum(axis=3).sum(axis=2)
    logical_and = np.logical_and(predicted_masks, true_masks).sum(axis=3).sum(axis=2) + 1e-6
    pairwise_iou = logical_and / logical_or
    
    thresholded_ious = []
    for threshold in np.arange(start=0.5, stop=1, step=0.05):
        n_intersects = np.sum(pairwise_iou >= threshold)
        thresholded_iou = n_intersects / (n_true + n_predicted - n_intersects)
        thresholded_ious.append(thresholded_iou)
    return np.mean(thresholded_ious)