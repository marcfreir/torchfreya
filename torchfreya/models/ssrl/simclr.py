from typing import Any, Callable, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.functional import normalize
import lightning as L
from torchvision import transforms
from torch.optim.optimizer import Optimizer, required
from PIL import Image
from torchfreya.augmentations.symbiosis_augmentations_v1 import RandomAugmentations
from typing import Set, Optional
from torchfreya.optimizers.lars import LARS


# ------------------------------------------------------------------------------
# Projection Head for SimCLR
# ------------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the projection head."""
        return self.layers(x)


# ------------------------------------------------------------------------------
# SimCLR Lightning Module
# ------------------------------------------------------------------------------
class SimCLR(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        projector_dim: int,
        hidden_dim: int,
        output_dim: int,
        temperature: float = 0.5,
        lr: float = 1e-3,
        test_metric: Optional[Callable] = None,
        num_classes: Optional[int] = None,
        augmentations: Optional[Any] = None,
    ):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = ProjectionHead(projector_dim, hidden_dim, output_dim)
        self.temperature = temperature
        self.lr = lr
        self.test_metric = test_metric
        self.num_classes = num_classes
        self.augmentations = (
            augmentations if augmentations is not None else RandomAugmentations()
        )

        self.to_pil = transforms.ToPILImage()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        projections = self.projector(flattened)
        return normalize(projections, dim=1)

    def _augment_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate two augmented views using RandomAugmentations."""
        batch_size = x.size(0)
        aug1, aug2 = [], []

        for i in range(batch_size):
            img_tensor = x[i]
            # Convert tensor to PIL Image (ensure CPU for conversion)
            pil_img = self.to_pil(img_tensor.cpu())

            # Apply augmentation (returns tensor directly)
            aug1_tensor = self.augmentations(pil_img)[0].to(
                x.device
            )  # [0] gets tensor from tuple
            aug2_tensor = self.augmentations(pil_img)[0].to(x.device)

            aug1.append(aug1_tensor)
            aug2.append(aug2_tensor)

        return torch.stack(aug1), torch.stack(aug2)

    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.size(0)
        projections = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.mm(projections, projections.T) / self.temperature
        labels = torch.cat(
            [torch.arange(N, device=z1.device) + N, torch.arange(N, device=z1.device)]
        )
        mask = torch.eye(2 * N, dtype=torch.bool, device=z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)
        return nn.CrossEntropyLoss()(similarity_matrix, labels)

    def _shared_step(self, batch: Any, prefix: str) -> torch.Tensor:
        """Shared logic for train/val/test steps."""
        x, _ = batch
        x1, x2 = self._augment_batch(x)
        z1 = self(x1)
        z2 = self(x2)
        loss = self.nt_xent_loss(z1, z2)
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        # Identify parameters to exclude (BatchNorm/bias)
        exclude_set = set()
        for name, param in self.named_parameters():
            if "bn" in name or "bias" in name:
                param.param_name = name  # Required for LARS exclusion
                exclude_set.add(name)

        return LARS(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-6,  # Default from paper
            eta=0.001,  # Trust coefficient (paper value)
            epsilon=1e-8,  # Numerical stability
            exclude_from_layer_adaptation=exclude_set,
        )
