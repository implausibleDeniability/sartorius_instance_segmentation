import numpy as np


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
        for j in range(i+1, n_masks):
            remove_overlapping_pixels_two_masks(masks[i], masks[j])
    return masks
        
                       