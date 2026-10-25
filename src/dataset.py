from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


LESIONS = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
}


class IDRiDSegmentationDataset(Dataset):
    def __init__(
        self,
        metadata_csv,
        dataset_root,
        split,
        transform=None,
    ):
        self.metadata = pd.read_csv(metadata_csv)
        self.metadata = self.metadata[
            self.metadata["split"] == split
        ].reset_index(drop=True)

        self.dataset_root = Path(dataset_root)
        self.transform = transform

        if split == "test":
            set_directory = "b. Testing Set"
        else:
            set_directory = "a. Training Set"

        self.image_directory = (
            self.dataset_root
            / "1. Original Images"
            / set_directory
        )

        self.mask_root = (
            self.dataset_root
            / "2. All Segmentation Groundtruths"
            / set_directory
        )

        if len(self.metadata) == 0:
            raise ValueError(f"No samples found for split: {split}")

    def __len__(self):
        return len(self.metadata)

    def _load_image(self, image_id):
        image_path = self.image_directory / f"{image_id}.jpg"

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_masks(self, image_id, image_shape):
        height, width = image_shape[:2]
        masks = []

        for lesion_code, directory_name in LESIONS.items():
            mask_path = (
                self.mask_root
                / directory_name
                / f"{image_id}_{lesion_code}.tif"
            )

            if mask_path.exists():
                mask = cv2.imread(
                    str(mask_path),
                    cv2.IMREAD_GRAYSCALE,
                )

                if mask is None:
                    raise ValueError(
                        f"Could not read mask: {mask_path}"
                    )

                mask = (mask > 0).astype(np.float32)
            else:
                mask = np.zeros(
                    (height, width),
                    dtype=np.float32,
                )

            masks.append(mask)

        return np.stack(masks, axis=-1)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        image_id = row["image_id"]

        image = self._load_image(image_id)
        mask = self._load_masks(image_id, image.shape)

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image = transformed["image"]
            mask = transformed["mask"].float()

        if image.shape[0] != 3:
            raise ValueError(
                f"Expected 3 image channels, got {image.shape}"
            )

        if mask.shape[0] != 4:
            raise ValueError(
                f"Expected 4 mask channels, got {mask.shape}"
            )

        return image, mask, image_id