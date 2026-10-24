import torch.nn as nn
import segmentation_models_pytorch as smp


class CombinedSegmentationLoss(nn.Module):
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

        total_loss = 0.5 * bce_loss + 0.5 * dice_loss

        return total_loss