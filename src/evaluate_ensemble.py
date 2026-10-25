import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import IDRiDSegmentationDataset
from src.evaluate import calculate_metrics, load_model
from src.transforms import get_validation_transforms


THRESHOLDS = [0.38, 0.48, 0.90, 0.32]


def create_loader(
    dataset_root,
    metadata_csv,
    split,
    image_size,
    batch_size,
    num_workers,
    device,
):
    dataset = IDRiDSegmentationDataset(
        metadata_csv=metadata_csv,
        dataset_root=dataset_root,
        split=split,
        transform=get_validation_transforms(image_size),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


@torch.inference_mode()
def predict_checkpoint(
    checkpoint_path,
    loader,
    device,
    channels,
    output_size=(1024, 1024),
    collect_masks=False,
):
    model = load_model(checkpoint_path, device)

    collected_probabilities = []
    collected_masks = []
    collected_ids = []

    for images, masks, image_ids in tqdm(
        loader,
        desc=Path(checkpoint_path).stem,
    ):
        images = images.to(device)

        with torch.amp.autocast(
            "cuda",
            enabled=device.type == "cuda",
        ):
            normal_probabilities = torch.sigmoid(model(images))

            flipped_images = torch.flip(images, dims=[3])
            flipped_probabilities = torch.sigmoid(
                model(flipped_images)
            )
            flipped_probabilities = torch.flip(
                flipped_probabilities,
                dims=[3],
            )

            probabilities = (
                normal_probabilities + flipped_probabilities
            ) / 2.0

            if probabilities.shape[-2:] != output_size:
                probabilities = F.interpolate(
                    probabilities,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )

        collected_probabilities.append(
            probabilities[:, channels].cpu()
        )
        collected_ids.extend(image_ids)

        if collect_masks:
            if masks.shape[-2:] != output_size:
                masks = F.interpolate(
                    masks.float(),
                    size=output_size,
                    mode="nearest",
                )

            collected_masks.append(masks.cpu())

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    probabilities = torch.cat(collected_probabilities)

    if collect_masks:
        masks = torch.cat(collected_masks)
    else:
        masks = None

    return probabilities, masks, collected_ids


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_directory = Path(args.checkpoint_dir)

    high_resolution_checkpoint = (
        checkpoint_directory / "lesion_aware_1024.pt"
    )
    lesion_aware_checkpoint = (
        checkpoint_directory / "lesion_aware_768.pt"
    )
    baseline_checkpoint = (
        checkpoint_directory / "pretrained_baseline_768.pt"
    )

    checkpoint_paths = [
        high_resolution_checkpoint,
        lesion_aware_checkpoint,
        baseline_checkpoint,
    ]

    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint_path}"
            )

    print("Device:", device)
    print("Evaluation resolution: 1024 x 1024")
    print("TTA: horizontal-flip averaging")
    print("Thresholds:", THRESHOLDS)

    loader_1024 = create_loader(
        dataset_root=args.dataset_root,
        metadata_csv=args.metadata_csv,
        split=args.split,
        image_size=1024,
        batch_size=args.batch_size_1024,
        num_workers=args.num_workers,
        device=device,
    )

    loader_768 = create_loader(
        dataset_root=args.dataset_root,
        metadata_csv=args.metadata_csv,
        split=args.split,
        image_size=768,
        batch_size=args.batch_size_768,
        num_workers=args.num_workers,
        device=device,
    )

    # High-resolution model:
    # local channels 0=MA, 1=HE, 2=SE.
    highres_predictions, masks, highres_ids = predict_checkpoint(
        checkpoint_path=high_resolution_checkpoint,
        loader=loader_1024,
        device=device,
        channels=[0, 1, 3],
        collect_masks=True,
    )

    # Lesion-aware 768 model:
    # local channels 0=MA, 1=SE.
    lesion_predictions, _, lesion_ids = predict_checkpoint(
        checkpoint_path=lesion_aware_checkpoint,
        loader=loader_768,
        device=device,
        channels=[0, 3],
    )

    # Pretrained baseline:
    # local channels 0=HE, 1=EX.
    baseline_predictions, _, baseline_ids = predict_checkpoint(
        checkpoint_path=baseline_checkpoint,
        loader=loader_768,
        device=device,
        channels=[1, 2],
    )

    if not (
        highres_ids == lesion_ids == baseline_ids
    ):
        raise ValueError(
            "Image order differs between model predictions."
        )

    final_probabilities = torch.empty_like(masks)

    # Microaneurysms:
    # average high-resolution and lesion-aware 768 models.
    final_probabilities[:, 0] = (
        highres_predictions[:, 0]
        + lesion_predictions[:, 0]
    ) / 2.0

    # Hemorrhages:
    # average high-resolution and pretrained baseline models.
    final_probabilities[:, 1] = (
        highres_predictions[:, 1]
        + baseline_predictions[:, 0]
    ) / 2.0

    # Hard exudates:
    # use only the pretrained baseline.
    final_probabilities[:, 2] = baseline_predictions[:, 1]

    # Soft exudates:
    # average high-resolution and lesion-aware 768 models.
    final_probabilities[:, 3] = (
        highres_predictions[:, 2]
        + lesion_predictions[:, 1]
    ) / 2.0

    metrics = calculate_metrics(
        probabilities=final_probabilities,
        masks=masks,
        thresholds=THRESHOLDS,
    )

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(
        output_directory / "test_metrics.csv",
        index=False,
    )

    summary = {
        "system": "final_multiscale_system",
        "test_images": len(highres_ids),
        "evaluation_resolution": [1024, 1024],
        "tta": "horizontal_flip_average",
        "thresholds_selected_on": "validation_set",
        "thresholds": {
            "microaneurysms": 0.38,
            "hemorrhages": 0.48,
            "hard_exudates": 0.90,
            "soft_exudates": 0.32,
        },
        "mean_dice": float(metrics["dice"].mean()),
        "mean_iou": float(metrics["iou"].mean()),
        "mean_aupr": float(metrics["aupr"].mean()),
    }

    with open(
        output_directory / "test_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nFinal multiscale results:")
    print(metrics.to_string(index=False))

    print("\nMean Dice:", summary["mean_dice"])
    print("Mean IoU:", summary["mean_iou"])
    print("Mean AUPR:", summary["mean_aupr"])
    print("Results saved to:", output_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the final multiscale IDRiD segmentation system."
        )
    )

    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/final_evaluation")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size-768", type=int, default=2)
    parser.add_argument("--batch-size-1024", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)

    main(parser.parse_args())