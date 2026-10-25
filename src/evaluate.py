import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import IDRiDSegmentationDataset
from src.model import create_model
from src.transforms import get_validation_transforms


LESION_NAMES = [
    "microaneurysms",
    "hemorrhages",
    "hard_exudates",
    "soft_exudates",
]


def load_model(checkpoint_path, device):
    model = create_model(pretrained=False).to(device)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.inference_mode()
def collect_predictions(model, loader, device, use_tta):
    all_probabilities = []
    all_masks = []
    all_image_ids = []

    for images, masks, image_ids in tqdm(
        loader,
        desc="Evaluating",
    ):
        images = images.to(device)

        with torch.amp.autocast(
            "cuda",
            enabled=device.type == "cuda",
        ):
            probabilities = torch.sigmoid(model(images))

            if use_tta:
                flipped_images = torch.flip(images, dims=[3])

                flipped_probabilities = torch.sigmoid(
                    model(flipped_images)
                )

                flipped_probabilities = torch.flip(
                    flipped_probabilities,
                    dims=[3],
                )

                probabilities = (
                    probabilities + flipped_probabilities
                ) / 2.0

        all_probabilities.append(probabilities.cpu())
        all_masks.append(masks.cpu())
        all_image_ids.extend(image_ids)

    return (
        torch.cat(all_probabilities),
        torch.cat(all_masks),
        all_image_ids,
    )


def calculate_metrics(probabilities, masks, thresholds):
    results = []

    probabilities = probabilities.numpy()
    masks = masks.numpy()

    for channel, lesion_name in enumerate(LESION_NAMES):
        threshold = thresholds[channel]

        lesion_probabilities = probabilities[:, channel].reshape(-1)
        lesion_targets = masks[:, channel].reshape(-1).astype(np.uint8)

        lesion_predictions = (
            lesion_probabilities >= threshold
        ).astype(np.uint8)

        true_positive = np.logical_and(
            lesion_predictions == 1,
            lesion_targets == 1,
        ).sum()

        false_positive = np.logical_and(
            lesion_predictions == 1,
            lesion_targets == 0,
        ).sum()

        false_negative = np.logical_and(
            lesion_predictions == 0,
            lesion_targets == 1,
        ).sum()

        epsilon = 1e-8

        dice = (
            2 * true_positive
            / (
                2 * true_positive
                + false_positive
                + false_negative
                + epsilon
            )
        )

        iou = (
            true_positive
            / (
                true_positive
                + false_positive
                + false_negative
                + epsilon
            )
        )

        precision = (
            true_positive
            / (true_positive + false_positive + epsilon)
        )

        recall = (
            true_positive
            / (true_positive + false_negative + epsilon)
        )

        if lesion_targets.sum() > 0:
            aupr = average_precision_score(
                lesion_targets,
                lesion_probabilities,
            )
        else:
            aupr = float("nan")

        results.append(
            {
                "lesion": lesion_name,
                "threshold": threshold,
                "dice": float(dice),
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "aupr": float(aupr),
            }
        )

    return pd.DataFrame(results)


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("TTA:", args.tta)
    print("Thresholds:", args.thresholds)

    dataset = IDRiDSegmentationDataset(
        metadata_csv=args.metadata_csv,
        dataset_root=args.dataset_root,
        split=args.split,
        transform=get_validation_transforms(args.image_size),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    probabilities, masks, image_ids = collect_predictions(
        model=model,
        loader=loader,
        device=device,
        use_tta=args.tta,
    )

    metrics = calculate_metrics(
        probabilities=probabilities,
        masks=masks,
        thresholds=args.thresholds,
    )

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(
        output_directory / "metrics.csv",
        index=False,
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "image_size": args.image_size,
        "tta": args.tta,
        "thresholds": args.thresholds,
        "number_of_images": len(image_ids),
        "mean_dice": float(metrics["dice"].mean()),
        "mean_iou": float(metrics["iou"].mean()),
        "mean_aupr": float(metrics["aupr"].mean()),
    }

    with open(
        output_directory / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nPer-lesion results:")
    print(metrics.to_string(index=False))

    print("\nMean Dice:", summary["mean_dice"])
    print("Mean IoU:", summary["mean_iou"])
    print("Mean AUPR:", summary["mean_aupr"])
    print("Saved to:", output_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate an IDRiD lesion segmentation checkpoint."
    )

    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=4,
        default=[0.5, 0.5, 0.5, 0.5],
        metavar=("MA", "HE", "EX", "SE"),
    )

    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use horizontal-flip test-time augmentation.",
    )

    main(parser.parse_args())