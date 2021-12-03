import os
import numpy as np
from dotenv import load_dotenv

def annotation2mask(annotation: str) -> np.ndarray:
    """Converts annotation string of the image 520x704 to np.ndarray 520x704"""
    # -1 because they numerate from 1, not from 0
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