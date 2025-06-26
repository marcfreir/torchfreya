from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class MultiClassDiceCELoss(nn.Module):
    def __init__(self, weight_ce: float = 1.0, weight_dice: float = 1.0) -> None:
        """Combined Dice and Cross-Entropy loss for multi-class segmentation.

        Combines Dice loss for handling class imbalance and Cross-Entropy loss for
        pixel-wise classification stability, improving segmentation performance.

        Parameters
        ----------
        weight_ce : float, optional
            Weight for Cross-Entropy loss component (default is 1.0).
        weight_dice : float, optional
            Weight for Dice loss component (default is 1.0).
        """

        super(MultiClassDiceCELoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(
        self, inputs: Union[torch.Tensor, List[torch.Tensor]], targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate combined Dice and Cross-Entropy loss.

        Parameters
        ----------
        inputs : torch.Tensor or list of torch.Tensor
            Model predictions. If a list, assumes deep supervision with multiple outputs.
        targets : torch.Tensor
            Ground truth segmentation masks of shape (batch_size, H, W) or
            (batch_size, 1, H, W).
        """

        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if isinstance(inputs, list):
            loss = 0
            for logits in inputs:
                ce_loss = nn.CrossEntropyLoss()(logits, targets.long())
                probs = F.softmax(logits, dim=1)
                dice_loss = self._dice_loss(probs, targets)
                combined_loss = self.weight_ce * ce_loss + self.weight_dice * dice_loss
                loss += combined_loss
            return loss / len(inputs)
        else:
            ce_loss = nn.CrossEntropyLoss()(inputs, targets.long())
            probs = F.softmax(inputs, dim=1)
            dice_loss = self._dice_loss(probs, targets)
            return self.weight_ce * ce_loss + self.weight_dice * dice_loss

    def _dice_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Dice loss for all classes.

        Computes Dice loss as 1 - (2 * intersection) / (pred + target) averaged
        over all classes.

        Parameters
        ----------
        probs : torch.Tensor
            Softmax probabilities of shape (batch_size, num_classes, H, W).
        targets : torch.Tensor
            Ground truth labels of shape (batch_size, H, W).
        """

        dice_loss = 0
        num_classes = probs.shape[1]
        for cls in range(num_classes):
            pred_cls = probs[:, cls]
            target_cls = (targets == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() + 1e-8
            dice_score = (2.0 * intersection) / union
            dice_loss += 1 - dice_score
        return dice_loss / num_classes
