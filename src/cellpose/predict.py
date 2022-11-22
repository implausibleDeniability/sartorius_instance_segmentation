import numpy as np
from cellpose import models, io, plot
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import glob


def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def generate_predictions_df(test_dir: Path, pretrained_model: Path) -> pd.DataFrame:
    test_files = glob.glob(f'{test_dir.__str__()}/*_img.tif')
    print(f"Number of test files: {len(test_files)}")
    model = models.CellposeModel(gpu=True, pretrained_model=str(pretrained_model))

    ids, masks = [], []
    for fn in tqdm(test_files):
        id_ = fn.split('/')[-1].replace('_img.tif', '')
        preds, flows, _ = model.eval(io.imread(fn), diameter=19, channels=[0, 0], augment=True, resample=True)
        for i in range(1, preds.max() + 1):
            ids.append(id_)
            masks.append(rle_encode(preds == i))

    return pd.DataFrame({'id': ids, 'predicted': masks})
