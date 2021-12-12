import albumentations as A
from albumentations.pytorch import ToTensorV2

eval_transform = A.Compose([
    A.Normalize(mean=(0.485,), std=(0.229,)),
    ToTensorV2(),
])
