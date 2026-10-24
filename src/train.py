import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import IDRiDSegmentationDataset
from src.losses import CombinedSegmentationLoss
from src.model import create_model
from src.transforms import (
    get_training_transforms,
    get_validation_transforms,
)


LESION_NAMES = [
    "microaneurysms",
    "hemorrhages",
    "hard_exudates",
    "soft_exudates",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_amp_context(use_amp):
    if use_amp:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp,
):
    model.train()
    total_loss = 0.0

    progress = tqdm(loader, desc="Training", leave=False)

    for images, masks, _ in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with get_amp_context(use_amp):
            predictions = model(images)
            loss = criterion(predictions, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    use_amp,
    threshold=0.5,
):
    model.eval()

    total_loss = 0.0
    intersection = torch.zeros(4, device=device)
    predicted_pixels = torch.zeros(4, device=device)
    target_pixels = torch.zeros(4, device=device)

    progress = tqdm(loader, desc="Validation", leave=False)

    for images, masks, _ in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with get_amp_context(use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).float()

        intersection += (
            predictions * masks
        ).sum(dim=(0, 2, 3))

        predicted_pixels += predictions.sum(dim=(0, 2, 3))
        target_pixels += masks.sum(dim=(0, 2, 3))

        total_loss += loss.item() * images.size(0)

    dice_scores = (
        2.0 * intersection + 1e-7
    ) / (
        predicted_pixels + target_pixels + 1e-7
    )

    return (
        total_loss / len(loader.dataset),
        dice_scores.cpu().numpy(),
    )


def main(args):
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    use_amp = device.type == "cuda"

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    with open(output_directory / "config.json", "w") as file:
        json.dump(vars(args), file, indent=2)

    train_dataset = IDRiDSegmentationDataset(
        metadata_csv=args.metadata_csv,
        dataset_root=args.dataset_root,
        split="train",
        transform=get_training_transforms(args.image_size),
    )

    validation_dataset = IDRiDSegmentationDataset(
        metadata_csv=args.metadata_csv,
        dataset_root=args.dataset_root,
        split="validation",
        transform=get_validation_transforms(args.image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(
        pretrained=args.pretrained
    ).to(device)

    criterion = CombinedSegmentationLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        validation_loss, dice_scores = validate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        mean_dice = float(np.mean(dice_scores))
        learning_rate = optimizer.param_groups[0]["lr"]

        result = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "mean_dice": mean_dice,
        }

        for lesion_name, dice_score in zip(
            LESION_NAMES,
            dice_scores,
        ):
            result[f"{lesion_name}_dice"] = float(dice_score)

        history.append(result)

        pd.DataFrame(history).to_csv(
            output_directory / "training_history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss: {train_loss:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"mean Dice: {mean_dice:.4f}"
        )

        for lesion_name, dice_score in zip(
            LESION_NAMES,
            dice_scores,
        ):
            print(f"  {lesion_name}: {dice_score:.4f}")

        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                    "dice_scores": dice_scores,
                    "image_size": args.image_size,
                    "pretrained": args.pretrained,
                    "lesion_names": LESION_NAMES,
                },
                output_directory / "best_model.pt",
            )

            print("  Saved new best model.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print("Early stopping.")
            break

    print("Training complete.")
    print("Best validation loss:", best_validation_loss)
    print("Outputs:", output_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/baseline",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
    )

    main(parser.parse_args())