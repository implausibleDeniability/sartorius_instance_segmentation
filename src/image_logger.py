import numpy as np
import torch
import wandb
from pytorch_toolbelt.utils import to_numpy
from skimage import color
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import images2device
from src.visualization import tensor_to_image


def apply_mask(image, mask, _color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * _color[c] * 255,
                                  image[:, :, c])
    return image


class ImageLogger:
    """Logs random n images from dataset
    """

    def __init__(self,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 device: str,
                 n_images: int):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device

        self.train_indices = np.random.choice(list(range(len(train_dataset))), size=n_images)
        self.val_indices = np.random.choice(list(range(len(val_dataset))), size=n_images)

        self.true_color = (1, 1, 0)
        self.pred_color = (1, 0, 0)

    def make_predictions(self, model, indices, dataset):
        log_images = []
        for ii in tqdm(indices, desc="Logging train/val images"):
            image, target = dataset[ii]
            images = images2device([image], self.device)

            with torch.no_grad():
                pred = model.forward(images)[0]

            image = tensor_to_image(image).squeeze()
            image = color.gray2rgb(image)

            pred_masks = to_numpy(pred['masks']).squeeze()
            pred_masks = (pred_masks > 0.5).astype(int)
            true_masks = to_numpy(target['masks']).squeeze()

            for mask in true_masks:
                image = apply_mask(image, mask, self.true_color)

            for mask in pred_masks:
                image = apply_mask(image, mask, self.pred_color)

            log_images.append(image)

        return np.concatenate(log_images, axis=1)

    def log_images(self, model: nn.Module):
        model.eval()

        train_samples = self.make_predictions(model, self.train_indices, self.train_dataset)
        wandb.log({"train/predictions": wandb.Image(train_samples, caption="Red is predictions, yellow is labels")})

        val_samples = self.make_predictions(model, self.val_indices, self.val_dataset)
        wandb.log({"val/predictions": wandb.Image(val_samples, caption="Red is predictions, yellow is labels")})
