
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class CombinedSegmentationLoss(nn.Module):
    """Original baseline: BCE + Dice."""

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = smp.losses.DiceLoss(
            mode="multilabel",
            from_logits=True,
        )

    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)

        return 0.5 * bce_loss + 0.5 * dice_loss


class LesionAwareLoss(nn.Module):
    """
    Focal BCE + class-weighted Focal Tversky loss.

    The weights give additional importance to small and difficult
    lesions, particularly microaneurysms and hemorrhages.
    """

    def __init__(
        self,
        focal_weight=0.4,
        tversky_weight=0.6,
        gamma=2.0,
        tversky_alpha=0.4,
        tversky_beta=0.6,
        smooth=1.0,
    ):
        super().__init__()

        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.gamma = gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.smooth = smooth

        # Order: MA, HE, EX, SE
        self.register_buffer(
            "positive_alpha",
            torch.tensor([0.95, 0.90, 0.75, 0.80]),
        )

        self.register_buffer(
            "class_weights",
            torch.tensor([2.0, 1.5, 1.0, 1.25]),
        )

    def focal_bce(self, logits, targets):
        probabilities = torch.sigmoid(logits)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        probability_of_correct_class = (
            probabilities * targets
            + (1.0 - probabilities) * (1.0 - targets)
        )

        alpha = self.positive_alpha.view(1, -1, 1, 1)

        alpha_factor = (
            alpha * targets
            + (1.0 - alpha) * (1.0 - targets)
        )

        focal_term = (
            1.0 - probability_of_correct_class
        ).pow(self.gamma)

        focal_loss = alpha_factor * focal_term * bce

        focal_loss_per_class = focal_loss.mean(
            dim=(0, 2, 3)
        )

        return (
            focal_loss_per_class * self.class_weights
        ).sum() / self.class_weights.sum()

    def focal_tversky(self, logits, targets):
        probabilities = torch.sigmoid(logits)

        dimensions = (0, 2, 3)

        true_positive = (
            probabilities * targets
        ).sum(dim=dimensions)

        false_positive = (
            probabilities * (1.0 - targets)
        ).sum(dim=dimensions)

        false_negative = (
            (1.0 - probabilities) * targets
        ).sum(dim=dimensions)

        tversky_score = (
            true_positive + self.smooth
        ) / (
            true_positive
            + self.tversky_alpha * false_positive
            + self.tversky_beta * false_negative
            + self.smooth
        )

        focal_tversky_per_class = (
            1.0 - tversky_score
        ).pow(self.gamma)

        return (
            focal_tversky_per_class * self.class_weights
        ).sum() / self.class_weights.sum()

    def forward(self, predictions, targets):
        focal_loss = self.focal_bce(
            predictions,
            targets,
        )

        tversky_loss = self.focal_tversky(
            predictions,
            targets,
        )

        return (
            self.focal_weight * focal_loss
            + self.tversky_weight * tversky_loss
        )
