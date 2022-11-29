from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import cv2
import matplotlib.pyplot as plt

saving_dir = Path("figures/broken_masks")
saving_dir.mkdir(exist_ok=True, parents=True)


def rle2mask(rle, img_w, img_h):
    array = np.fromiter(rle.split(), dtype=np.uint)
    array = array.reshape((-1, 2)).T
    array[0] = array[0] - 1

    starts, lenghts = array
    mask_decompressed = np.concatenate([np.arange(s, s + l, dtype=np.uint) for s, l in zip(starts, lenghts)])

    msk_img = np.zeros(img_w * img_h, dtype=np.uint8)
    msk_img[mask_decompressed] = 1
    msk_img = msk_img.reshape((img_h, img_w))

    return msk_img


count = 0

TH = 40

data_root = Path("data")
df = pd.read_csv(str(data_root.joinpath("train.csv")))
img_dir = data_root.joinpath("train")

for idx, row in df.iterrows():
    if idx > 5000:
        break

    mask = rle2mask(row['annotation'], row['width'], row['height'])
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = contours[0][:, 0]
    diff = c - np.roll(c, 1, 0)
    targets = (diff[:, 1] == 0) & (np.abs(diff[:, 0]) >= TH)  # find horizontal lines longer than threshold

    if targets.sum() == 0:
        continue

    if np.all(c[targets][:, 1] == 0) or np.all(c[targets][:, 1] == 519):  # remove screen edge cases
        continue

    img_id = row["id"]
    img_path = img_dir.joinpath(f"{img_id}.png")
    img = cv2.imread(str(img_path), 0)
    # plt.figure(figsize=(16, 12))
    # plt.title(f"{img_id} - {idx}")
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5)
    plt.axis('off')
    plt.grid(visible=None)
    plt.savefig(saving_dir / f"{row['id']}_{idx}.png", bbox_inches='tight', pad_inches=0.0)

    count += 1
    if count >= 60:
        break

print("Finished!")
