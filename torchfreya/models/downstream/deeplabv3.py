from typing import Dict, Optional, Sequence, List, Tuple, Any, Union
from collections import OrderedDict
import torch
from torch import Tensor, nn, optim
import lightning as L
from torchmetrics import Metric
from torchvision.models.resnet import resnet101, ResNet101_Weights
import torch.nn.functional as F


class DeepLabV3(L.LightningModule):
    """
    Complete DeepLabV3 implementation following the original paper specifications.

    References
    ----------
    Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam.
    "Rethinking Atrous Convolution for Semantic Image Segmentation", 2017
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        pred_head: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        learning_rate: float = 0.007,
        num_classes: int = 21,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        output_stride: int = 16,
        multi_grid: tuple = (1, 2, 4),
        use_cascaded: bool = False,
        train_metrics: Optional[Dict[str, Metric]] = None,
        val_metrics: Optional[Dict[str, Metric]] = None,
        test_metrics: Optional[Dict[str, Metric]] = None,
        optimizer: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[type] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Initialize backbone and prediction head if not provided
        self.backbone = backbone or DeepLabV3Backbone(
            pretrained=pretrained,
            weights_path=weights_path,
            output_stride=output_stride,
            multi_grid=multi_grid,
            use_cascaded=use_cascaded,
        )

        self.pred_head = pred_head or DeepLabV3PredictionHead(
            num_classes=num_classes,
            output_stride=output_stride,
            use_cascaded=use_cascaded,
        )

        self.loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=255)

        # Store hyperparameters
        self.learning_rate = learning_rate
        self.output_stride = output_stride
        self.use_cascaded = use_cascaded
        self.output_shape = output_shape
        self.freeze_backbone = freeze_backbone

        # Metrics for different phases
        self.train_metrics = train_metrics or {}
        self.val_metrics = val_metrics or {}
        self.test_metrics = test_metrics or {}

        # Optimizer configuration
        self.optimizer_class = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.lr_scheduler_class = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        input_shape = self.output_shape or x.shape[-2:]

        # Forward through backbone
        features = self.backbone(x)

        # Forward through prediction head
        logits = self.pred_head(features)

        # Bilinear upsampling to original input resolution
        return F.interpolate(
            logits, size=input_shape, mode="bilinear", align_corners=False
        )

    def forward_multiscale(
        self, x: Tensor, scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    ) -> Tensor:
        """Multi-scale inference as described in the paper."""
        input_shape = x.shape[-2:]
        logits_sum = None

        for scale in scales:
            if scale != 1.0:
                h = int(input_shape[0] * scale)
                w = int(input_shape[1] * scale)
                x_scaled = F.interpolate(
                    x, size=(h, w), mode="bilinear", align_corners=False
                )
            else:
                x_scaled = x

            # Forward pass
            logits = self.forward(x_scaled)

            # Resize back to original size
            if scale != 1.0:
                logits = F.interpolate(
                    logits, size=input_shape, mode="bilinear", align_corners=False
                )

            # Horizontal flip augmentation
            x_flipped = torch.flip(x_scaled, dims=[3])
            logits_flipped = self.forward(x_flipped)
            logits_flipped = torch.flip(logits_flipped, dims=[3])

            if scale != 1.0:
                logits_flipped = F.interpolate(
                    logits_flipped,
                    size=input_shape,
                    mode="bilinear",
                    align_corners=False,
                )

            # Average original and flipped
            logits = (logits + logits_flipped) / 2

            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum += logits

        return logits_sum / len(scales)

    def _compute_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """Compute loss between predictions and targets."""
        return self.loss_fn(y_hat, y.squeeze(1).long())

    def _compute_metrics(
        self, y_hat: Tensor, y: Tensor, metrics: Dict[str, Metric]
    ) -> Dict[str, Tensor]:
        """Calculate metrics for the given predictions and targets."""
        if not metrics:
            return {}

        computed_metrics = {}
        for metric_name, metric in metrics.items():
            metric_value = metric.to(self.device)(y_hat, y)
            computed_metrics[metric_name] = metric_value

        return computed_metrics

    def _shared_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, stage: str
    ) -> Tensor:
        """Shared step for training, validation, and testing."""
        x, y = batch
        y_hat = self.forward(x)
        loss = self._compute_loss(y_hat, y)

        # Log loss
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Compute and log metrics
        if stage == "train":
            metrics = self._compute_metrics(y_hat, y, self.train_metrics)
        elif stage == "val":
            metrics = self._compute_metrics(y_hat, y, self.val_metrics)
        else:  # test
            metrics = self._compute_metrics(y_hat, y, self.test_metrics)

        for metric_name, metric_value in metrics.items():
            self.log(
                f"{stage}_{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Validation step."""
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Test step."""
        return self._shared_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Prediction step."""
        x, _ = batch
        y_hat = self.forward(x)
        return y_hat

    def freeze_backbone_weights(self):
        """Freeze backbone weights for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone_weights(self):
        """Unfreeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        Implements paper's training protocol with poly learning rate policy.
        """
        # Handle backbone freezing
        if self.freeze_backbone:
            self.freeze_backbone_weights()
        else:
            self.unfreeze_backbone_weights()

        # Always make prediction head trainable
        for param in self.pred_head.parameters():
            param.requires_grad = True

        # Separate backbone and head parameters for different learning rates
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.pred_head.parameters())

        if self.optimizer_class == optim.SGD:
            # Paper's configuration: different learning rates for backbone and head
            optimizer = self.optimizer_class(
                [
                    {
                        "params": backbone_params,
                        "lr": self.learning_rate * 0.1,
                    },  # Lower LR for pretrained backbone
                    {"params": head_params, "lr": self.learning_rate},
                ],
                momentum=0.9,
                weight_decay=5e-4,
                **self.optimizer_kwargs,
            )
        else:
            # For other optimizers, use single learning rate
            optimizer = self.optimizer_class(
                self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
            )

        if self.lr_scheduler_class is None:
            return optimizer

        # Configure scheduler
        scheduler = self.lr_scheduler_class(optimizer, **self.lr_scheduler_kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class DeepLabV3Backbone(nn.Module):
    """
    Complete ResNet-101 backbone with atrous convolution for DeepLabV3.
    Implements multi-grid method, cascaded modules, and proper output stride control.
    """

    def __init__(
        self,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        output_stride: int = 16,
        multi_grid: tuple = (1, 2, 4),
        use_cascaded: bool = False,
    ):
        super().__init__()

        self.output_stride = output_stride
        self.multi_grid = multi_grid
        self.use_cascaded = use_cascaded

        # Configure dilations based on output_stride
        if output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
        else:
            replace_stride_with_dilation = [False, False, False]

        # Use ResNet-101 as per paper
        if pretrained and weights_path is None:
            try:
                self.resnet = resnet101(
                    weights=ResNet101_Weights.IMAGENET1K_V1,
                    replace_stride_with_dilation=replace_stride_with_dilation,
                )
                print("Successfully loaded ImageNet pretrained ResNet-101")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                self.resnet = resnet101(
                    replace_stride_with_dilation=replace_stride_with_dilation
                )
        else:
            self.resnet = resnet101(
                replace_stride_with_dilation=replace_stride_with_dilation
            )
            if weights_path:
                self._load_custom_weights(weights_path)

        # Apply multi-grid to layer4
        self._apply_multi_grid()

        # Remove final layers not needed for dense prediction
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        # Add cascaded modules if requested
        if use_cascaded:
            self.cascaded_blocks = self._create_cascaded_blocks()

    def _apply_multi_grid(self):
        """Apply multi-grid method to the last ResNet block (layer4)."""
        if self.output_stride == 16:
            base_dilation = 2
        elif self.output_stride == 8:
            base_dilation = 4
        else:
            base_dilation = 1

        # Apply multi-grid rates to layer4 blocks
        for i, bottleneck in enumerate(self.resnet.layer4):
            if hasattr(bottleneck, "conv2"):
                grid_rate = self.multi_grid[i % len(self.multi_grid)]
                new_dilation = base_dilation * grid_rate
                bottleneck.conv2.dilation = (new_dilation, new_dilation)
                bottleneck.conv2.padding = (new_dilation, new_dilation)

    def _create_cascaded_blocks(self):
        """Create additional cascaded blocks as in paper."""
        cascaded_blocks = nn.ModuleList()

        for block_idx in range(3):  # block5, block6, block7
            block = self._make_cascaded_block(2048, self.multi_grid)
            cascaded_blocks.append(block)

        return cascaded_blocks

    def _make_cascaded_block(self, channels: int, multi_grid: tuple):
        """Create a single cascaded block with multi-grid."""
        if self.output_stride == 16:
            base_dilation = 2
        elif self.output_stride == 8:
            base_dilation = 4
        else:
            base_dilation = 1

        layers = []
        for i, rate in enumerate(multi_grid):
            dilation = base_dilation * rate

            block = nn.Sequential(
                nn.Conv2d(channels, channels // 4, 1, bias=False),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    channels // 4,
                    channels // 4,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            )
            layers.append(block)

        return nn.ModuleList(layers)

    def _load_custom_weights(self, weights_path: str):
        """Load weights from custom path."""
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

            # Filter out final classification layer
            filtered_state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith("fc.")
            }

            missing_keys, unexpected_keys = self.resnet.load_state_dict(
                filtered_state_dict, strict=False
            )
            print(f"Loaded custom weights from {weights_path}")

        except Exception as e:
            print(f"Error loading custom weights: {e}")

    def forward(self, x):
        # Forward through ResNet backbone
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # Apply cascaded blocks if enabled
        if self.use_cascaded and hasattr(self, "cascaded_blocks"):
            for cascaded_block in self.cascaded_blocks:
                residual = x
                for sub_block in cascaded_block:
                    x = sub_block(x)
                x = F.relu(x + residual)

        return x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module with fixes for small feature maps.
    """

    def __init__(
        self, in_channels: int = 2048, out_channels: int = 256, output_stride: int = 16
    ):
        super().__init__()

        # Configure atrous rates based on output stride
        if output_stride == 16:
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]
        else:
            atrous_rates = [6, 12, 18]

        # ASPP branches
        self.aspp_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.aspp_3x3_1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=atrous_rates[0],
                dilation=atrous_rates[0],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.aspp_3x3_2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=atrous_rates[1],
                dilation=atrous_rates[1],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.aspp_3x3_3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=atrous_rates[2],
                dilation=atrous_rates[2],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        h, w = x.shape[2:]

        # Apply all ASPP branches
        feat1 = self.aspp_1x1(x)
        feat2 = self.aspp_3x3_1(x)
        feat3 = self.aspp_3x3_2(x)
        feat4 = self.aspp_3x3_3(x)

        # Global average pooling branch
        feat5 = self.global_avg_pool(x)
        if h > 1 and w > 1:
            feat5 = F.interpolate(
                feat5, size=(h, w), mode="bilinear", align_corners=False
            )
        else:
            feat5 = feat5.expand(-1, -1, h, w)

        # Concatenate all features
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.project(out)

        return out


class DeepLabV3PredictionHead(nn.Module):
    """
    DeepLabV3 prediction head with ASPP or simple classifier for cascaded version.
    """

    def __init__(
        self,
        in_channels: int = 2048,
        num_classes: int = 21,
        output_stride: int = 16,
        use_cascaded: bool = False,
    ):
        super().__init__()

        self.use_cascaded = use_cascaded

        if not use_cascaded:
            # Standard ASPP-based head
            self.aspp = ASPP(
                in_channels=in_channels, out_channels=256, output_stride=output_stride
            )

            # Final classifier
            self.classifier = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(256, num_classes, 1),
            )
        else:
            # Simple classifier for cascaded version
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(256, num_classes, 1),
            )

    def forward(self, x):
        if not self.use_cascaded:
            x = self.aspp(x)
        x = self.classifier(x)
        return x


# Factory functions for easy model creation
def create_deeplabv3_aspp_model(
    num_classes: int = 21,
    output_stride: int = 16,
    pretrained: bool = True,
    multi_grid: tuple = (1, 2, 4),
    learning_rate: float = 0.007,
) -> DeepLabV3:
    """Create DeepLabV3 model with ASPP (paper's final model)."""

    return DeepLabV3(
        num_classes=num_classes,
        output_stride=output_stride,
        pretrained=pretrained,
        multi_grid=multi_grid,
        use_cascaded=False,
        learning_rate=learning_rate,
    )


def create_deeplabv3_cascaded_model(
    num_classes: int = 21,
    output_stride: int = 16,
    pretrained: bool = True,
    multi_grid: tuple = (1, 2, 4),
    learning_rate: float = 0.007,
) -> DeepLabV3:
    """Create DeepLabV3 model with cascaded modules."""

    return DeepLabV3(
        num_classes=num_classes,
        output_stride=output_stride,
        pretrained=pretrained,
        multi_grid=multi_grid,
        use_cascaded=True,
        learning_rate=learning_rate,
    )


def create_deeplabv3_for_custom_classes(
    num_classes: int = 6,
    output_stride: int = 16,
    pretrained: bool = True,
    multi_grid: tuple = (1, 2, 4),
    learning_rate: float = 0.007,
) -> DeepLabV3:
    """
    Create DeepLabV3 model for custom number of classes.

    Args:
        num_classes: Number of segmentation classes (including background if applicable)
        output_stride: Output stride (8 or 16)
        pretrained: Whether to use ImageNet pretrained backbone
        multi_grid: Multi-grid configuration for layer4
        learning_rate: Initial learning rate

    Returns:
        Configured DeepLabV3 model
    """

    return DeepLabV3(
        num_classes=num_classes,
        output_stride=output_stride,
        pretrained=pretrained,
        multi_grid=multi_grid,
        use_cascaded=False,
        learning_rate=learning_rate,
    )
