import os

import pandas as pd
import torch
from cellpose.models import CellposeModel
from pathlib import Path
from cellpose import io
import cv2
from src.utils import annotation2mask
from torchvision.utils import draw_segmentation_masks

os.environ['IMAGE_HEIGHT'] = str(520)
os.environ['IMAGE_WIDTH'] = str(704)


def draw_beautiful_image(image_rgb, targets, predictions):
    image_torch = torch.as_tensor(image_rgb, dtype=torch.uint8).permute(2, 0, 1)
    predictions = torch.as_tensor(predictions, dtype=torch.bool)
    targets = torch.as_tensor(targets, dtype=torch.bool)

    pred = draw_segmentation_masks(
        image=image_torch,
        masks=predictions,
        alpha=0.5
    ).permute(1, 2, 0).numpy()

    target = draw_segmentation_masks(
        image=image_torch,
        masks=targets,
        alpha=0.5
    ).permute(1, 2, 0).numpy()

    return pred, target


def generate_images(
        model: CellposeModel,
        image_dir: Path,
        anno_df: pd.DataFrame,
        saving_path: Path
):
    cellpose_pred_path = saving_path / 'cellpose_mask_predictions'
    cellpose_pred_path.mkdir(exist_ok=True, parents=True)

    target_pred_path = saving_path / 'test_mask_visualisation'
    target_pred_path.mkdir(exist_ok=True, parents=True)

    for image_path in image_dir.glob("**/*"):
        print(f"Processing: {image_path.resolve()}")
        image_id = image_path.stem

        # reading image
        image_rgb = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        # generating predictions
        image = io.imread(str(image_path))

        masks, _, _ = model.eval(image, diameter=19, channels=[0, 0], augment=True, resample=True)

        predictions = []

        for ii in range(1, masks.max() + 1):
            mask = masks == ii
            predictions.append(mask)

        # generating ground truth
        targets = []

        annotations = anno_df[anno_df.id == image_id]
        for ii, row in annotations.iterrows():
            mask = annotation2mask(row.annotation)
            targets.append(mask)

        image_pred, image_target = draw_beautiful_image(image_rgb, targets, predictions)

        image_pred_path = cellpose_pred_path / image_path.name
        image_target_path = target_pred_path / image_path.name

        cv2.imwrite(str(image_pred_path), image_pred)
        cv2.imwrite(str(image_target_path), image_target)


def main():
    saving_path = Path("figures/")
    cellpose_path = Path("weights/cellpose_weights.ckpt")
    model = CellposeModel(
        pretrained_model=str(cellpose_path),
        gpu=True
    )

    test_path = Path("data/hidden_test")
    test_image_path = test_path / "images"
    test_anno_path = test_path / 'test.csv'

    test_anno_df = pd.read_csv(test_anno_path)

    generate_images(
        model=model,
        image_dir=test_image_path,
        anno_df=test_anno_df,
        saving_path=saving_path
    )


if __name__ == '__main__':
    main()
