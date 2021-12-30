import numpy as np


def compute_iou(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        y_true : true masks, np.ndarray with shape [N_MASKS, HEIGHT, WIDTH]
        y_pred : predicted masks, np.ndarray with shape [M_MASKS, HEIGHT, WIDTH]

    Returns:
        np array: IoU matrix, of size N_MASKS x M_MASKS.
    """

    true_objects = len(np.unique(y_true))
    pred_objects = len(np.unique(y_pred))

    # Compute intersection between all objects
    intersection = np.histogram2d(
        y_true.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(y_true, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
    iou = intersection / (union + 1e-6)
    iou = iou[1:, 1:] # exclude background
    return iou 


def precision_at(threshold: float, iou: np.ndarray):
    """
    Computes the precision at a given threshold.

    Args:
        threshold (float): Threshold.
        iou (np array [n_truths x n_preds]): IoU matrix.

    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) >= 1  # Correct objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    return tp, fp, fn


def flatten_masks(masks: np.ndarray):
    """Takes the array of binary masks, enumerates them and collect them in 2d mask"""
    pred = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.int64)
    for ii, mask in enumerate(masks):
        pred[mask == 1] = ii + 1

    return pred


def iou_map(true_masks: np.ndarray, pred_masks: np.ndarray, verbose: int=0) -> float:
    """
    Computes the metric for the competition.
    Masks contain the segmented pixels where each object has one value associated,
    and 0 is the background.

    Args:
        true_masks: true masks, np.ndarray with shape [N_MASKS, HEIGHT, WIDTH]
        pred_masks: predicted masks, np.ndarray with shape [M_MASKS, HEIGHT, WIDTH]
        verbose (int, optional): Whether to print infos. Defaults to 0.

    Returns:
        float: mAP.
    """
    truth = flatten_masks(true_masks)
    pred = flatten_masks(pred_masks)
    ious = [compute_iou(truth, pred)]


    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn

        p = tps / (tps + fps + fns)
        prec.append(p)

        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, p))

    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return np.mean(prec)
