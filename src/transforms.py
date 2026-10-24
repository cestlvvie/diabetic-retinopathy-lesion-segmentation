import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_training_transforms(image_size=768):
    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
                interpolation=cv2.INTER_AREA,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.HorizontalFlip(p=0.5),
            A.Rotate(
                limit=15,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5,
            ),
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
            ToTensorV2(transpose_mask=True),
        ]
    )


def get_validation_transforms(image_size=768):
    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
                interpolation=cv2.INTER_AREA,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
            ToTensorV2(transpose_mask=True),
        ]
    )