import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import lightning as L
from collections import deque
from typing import Optional, Any, Callable, Sequence, List, Tuple
import numpy as np
from torchfreya.transforms.symbiosis_augmentations4 import RandomAugmentations


class DynamicBoWPredictionHead(nn.Module):
    """Dynamic BoW prediction head that adapts to evolving vocabulary."""

    def __init__(self, input_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * input_dim

        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, vocabulary: torch.Tensor) -> torch.Tensor:
        # Normalize input vocabulary
        vocabulary_norm = F.normalize(vocabulary, p=2, dim=1)

        # Generate weights for each visual word
        weights = self.generator(vocabulary_norm)

        # Normalize output weights
        weights_norm = F.normalize(weights, p=2, dim=1)

        return weights_norm


class FeatureExtractor(nn.Module):
    """Feature extractor that hooks into ResNet backbone layers."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.features = {}
        self.hooks = []

        # Register hooks for conv4 (layer3) and conv5 (layer4)
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output

            return hook

        # Hook into layer3 (conv4) and layer4 (conv5)
        if hasattr(self.backbone.RN50model, "layer3"):
            hook = self.backbone.RN50model.layer3.register_forward_hook(
                get_activation("layer3")
            )
            self.hooks.append(hook)

        if hasattr(self.backbone.RN50model, "layer4"):
            hook = self.backbone.RN50model.layer4.register_forward_hook(
                get_activation("layer4")
            )
            self.hooks.append(hook)

    def forward(self, x):
        # Clear previous features
        self.features.clear()

        # Forward pass through backbone
        output = self.backbone(x)

        return {
            "output": output,
            "layer3": self.features.get("layer3"),  # conv4 (L1)
            "layer4": self.features.get("layer4"),  # conv5 (L)
        }

    def __del__(self):
        # Clean up hooks
        for hook in self.hooks:
            hook.remove()


class OBoW(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 2048,  # Feature dimension from backbone
        vocab_size: int = 8192,
        momentum: float = 0.99,
        lr: float = 0.05,
        weight_decay: float = 5e-4,
        temperature_base: float = 0.1,
        kappa: float = 5.0,
        use_multiscale: bool = True,
        augmentations: Optional[Callable] = None,
        num_crops_160: int = 1,
        num_crops_96: int = 2,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "augmentations"])

        # Model parameters
        self.vocab_size = vocab_size
        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature_base = temperature_base
        self.kappa = kappa
        self.use_multiscale = use_multiscale
        self.num_crops_160 = num_crops_160
        self.num_crops_96 = num_crops_96
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        # Student network with feature extractor
        self.student_extractor = FeatureExtractor(backbone)

        # Teacher network (momentum updated copy of student)
        teacher_backbone = copy.deepcopy(backbone)
        self.teacher_extractor = FeatureExtractor(teacher_backbone)

        # Freeze teacher initially
        for param in self.teacher_extractor.parameters():
            param.requires_grad = False

        # Get feature dimensions for different layers
        self.feature_dim_L = feature_dim  # Last layer (conv5/layer4)
        self.feature_dim_L1 = feature_dim // 2  # Penultimate layer (conv4/layer3)

        # Dynamic BoW prediction heads
        self.bow_head_L = DynamicBoWPredictionHead(self.feature_dim_L)
        if self.use_multiscale:
            self.bow_head_L1 = DynamicBoWPredictionHead(self.feature_dim_L1)

        # Vocabulary queues (FIFO) - using buffers
        self.register_buffer("vocab_L", torch.randn(vocab_size, self.feature_dim_L))
        if self.use_multiscale:
            self.register_buffer(
                "vocab_L1", torch.randn(vocab_size, self.feature_dim_L1)
            )

        # Normalize and scale initial vocabularies properly
        with torch.no_grad():
            self.vocab_L = (
                F.normalize(self.vocab_L, p=2, dim=1) * 0.1
            )  # Small initial scale
            if self.use_multiscale:
                self.vocab_L1 = F.normalize(self.vocab_L1, p=2, dim=1) * 0.1

        # Initialize EMA values more conservatively
        self.register_buffer("msd_ema_L", torch.tensor(0.5))
        if self.use_multiscale:
            self.register_buffer("msd_ema_L1", torch.tensor(0.5))

        # Vocabulary update counters
        self.vocab_update_counter = 0

        # Use provided augmentations or default to RandomAugmentations
        self.augmentations = (
            augmentations if augmentations is not None else RandomAugmentations()
        )

    def generate_multi_crops(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate multiple crops of different sizes."""
        batch_size = images.shape[0]
        device = images.device

        # Generate all crops at once for better efficiency
        all_crops_160 = []
        all_crops_96 = []

        for i in range(batch_size):
            img = images[i]

            # Generate crops of size 160x160
            for _ in range(self.num_crops_160):
                aug_img, _ = self.augmentations(img)
                aug_img = aug_img.to(device)
                # Resize to 160x160
                aug_img = F.interpolate(
                    aug_img.unsqueeze(0),
                    size=(160, 160),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                all_crops_160.append(aug_img)

            # Generate crops of size 96x96
            for _ in range(self.num_crops_96):
                aug_img, _ = self.augmentations(img)
                aug_img = aug_img.to(device)
                # Resize to 96x96
                aug_img = F.interpolate(
                    aug_img.unsqueeze(0),
                    size=(96, 96),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                all_crops_96.append(aug_img)

        # Stack crops into separate tensors (DON'T concatenate different sizes)
        crops_160 = torch.stack(all_crops_160) if all_crops_160 else None
        crops_96 = torch.stack(all_crops_96) if all_crops_96 else None

        return crops_160, crops_96

    def local_average_pooling(
        self, feature_map: torch.Tensor, kernel_size: int = 3
    ) -> torch.Tensor:
        """Apply local average pooling to feature map."""
        # feature_map: (B, C, H, W)
        pooled = F.avg_pool2d(feature_map, kernel_size=kernel_size, stride=1, padding=0)
        return pooled

    # Vocabulary update with sampling
    def update_vocabulary(self, teacher_features: torch.Tensor, layer: str = "L"):
        """Update vocabulary with better feature sampling."""
        # Apply pooling
        pooled_features = self.local_average_pooling(
            teacher_features, kernel_size=2
        )  # Smaller kernel
        B, C, H, W = pooled_features.shape

        # Flatten and normalize
        pooled_features = pooled_features.permute(0, 2, 3, 1).reshape(-1, C)
        pooled_features = F.normalize(pooled_features, p=2, dim=1)

        # Sampling strategy: sample multiple features per batch
        features_to_add = []
        samples_per_batch = min(3, H * W)  # Sample up to 3 features per image

        for b in range(B):
            start_idx = b * H * W
            end_idx = (b + 1) * H * W
            batch_features = pooled_features[start_idx:end_idx]

            if batch_features.shape[0] > 0:
                # Sample multiple features
                if batch_features.shape[0] >= samples_per_batch:
                    indices = torch.randperm(
                        batch_features.shape[0], device=batch_features.device
                    )[:samples_per_batch]
                    sampled_features = batch_features[indices]
                else:
                    sampled_features = batch_features

                features_to_add.extend([f for f in sampled_features])

        if not features_to_add:
            return

        new_features = torch.stack(features_to_add)

        # Update vocabulary with momentum (instead of hard replacement)
        if layer == "L":
            vocab_size = self.vocab_L.shape[0]
            update_size = min(
                new_features.shape[0], vocab_size // 4
            )  # Update only 1/4 of vocab per step

            if update_size > 0:
                # Random indices to update
                indices = torch.randperm(vocab_size, device=self.device)[:update_size]
                selected_features = new_features[:update_size]

                # Momentum update instead of hard replacement
                momentum = 0.1
                self.vocab_L[indices] = (1 - momentum) * self.vocab_L[
                    indices
                ] + momentum * selected_features
                self.vocab_L[indices] = F.normalize(self.vocab_L[indices], p=2, dim=1)

        elif layer == "L1" and self.use_multiscale:
            vocab_size = self.vocab_L1.shape[0]
            update_size = min(new_features.shape[0], vocab_size // 4)

            if update_size > 0:
                indices = torch.randperm(vocab_size, device=self.device)[:update_size]
                selected_features = new_features[:update_size]

                momentum = 0.1
                self.vocab_L1[indices] = (1 - momentum) * self.vocab_L1[
                    indices
                ] + momentum * selected_features
                self.vocab_L1[indices] = F.normalize(self.vocab_L1[indices], p=2, dim=1)

    # Temperature computation
    def compute_soft_assignment(
        self,
        features: torch.Tensor,
        vocabulary: torch.Tensor,
        msd_ema: torch.Tensor,
        layer: str = "L",
    ) -> torch.Tensor:
        """Compute soft assignment codes with better temperature handling."""
        B, C, H, W = features.shape
        K = vocabulary.shape[0]

        # Flatten and normalize
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        features_flat = F.normalize(features_flat, p=2, dim=1)

        # Compute similarities
        similarities = torch.mm(features_flat, vocabulary.T)
        distances = 1.0 - similarities

        # More stable temperature computation
        temperature = self.temperature_base * torch.clamp(msd_ema, min=0.01, max=2.0)

        # Soft assignment with clamped temperature
        soft_assignments = F.softmax(-distances / temperature, dim=1)
        soft_assignments = soft_assignments.reshape(B, H, W, K).permute(0, 3, 1, 2)

        # More stable MSD update
        with torch.no_grad():
            min_distances = distances.min(dim=1)[0]
            current_msd = min_distances.mean()
            # Clamp the current MSD to prevent extreme values
            current_msd = torch.clamp(current_msd, min=0.01, max=2.0)
            msd_ema.mul_(0.995).add_(current_msd, alpha=0.005)  # Slower EMA update

        return soft_assignments

    def generate_bow_target(
        self, teacher_features: torch.Tensor, layer: str = "L"
    ) -> torch.Tensor:
        """Generate BoW target from teacher features."""
        if layer == "L":
            vocabulary = self.vocab_L
            msd_ema = self.msd_ema_L
        elif layer == "L1":
            vocabulary = self.vocab_L1
            msd_ema = self.msd_ema_L1
        else:
            raise ValueError(f"Unknown layer: {layer}")

        # Compute soft assignments
        soft_assignments = self.compute_soft_assignment(
            teacher_features, vocabulary, msd_ema, layer
        )

        # Max pooling to get BoW representation
        bow_unnorm = F.max_pool2d(
            soft_assignments, kernel_size=soft_assignments.shape[-2:]
        )
        bow_unnorm = bow_unnorm.squeeze(-1).squeeze(-1)  # (B, K)

        # L1 normalization to get probability distribution
        bow_target = F.normalize(bow_unnorm, p=1, dim=1)

        return bow_target

    def predict_bow(
        self, student_features: torch.Tensor, layer: str = "L"
    ) -> torch.Tensor:
        """Predict BoW distribution using dynamic prediction head."""
        # Get current vocabulary
        if layer == "L":
            vocabulary = self.vocab_L
            bow_head = self.bow_head_L
        elif layer == "L1":
            vocabulary = self.vocab_L1
            bow_head = self.bow_head_L1
        else:
            raise ValueError(f"Unknown layer: {layer}")

        # Generate prediction weights
        prediction_weights = bow_head(vocabulary)  # (K, C)

        # Global average pooling for student features if they're feature maps
        if len(student_features.shape) == 4:
            student_features = F.adaptive_avg_pool2d(student_features, (1, 1)).flatten(
                1
            )

        # Normalize student features
        student_features = F.normalize(student_features, p=2, dim=1)

        # Compute similarities and apply softmax
        similarities = torch.matmul(student_features, prediction_weights.T)  # (B, K)
        bow_pred = F.softmax(self.kappa * similarities, dim=1)

        return bow_pred

    @torch.no_grad()
    def update_teacher(self):
        """Update teacher parameters using momentum."""
        for student_param, teacher_param in zip(
            self.student_extractor.parameters(), self.teacher_extractor.parameters()
        ):
            teacher_param.data.mul_(self.momentum).add_(
                student_param.data, alpha=1.0 - self.momentum
            )

    def forward(self, x: torch.Tensor):
        """Forward pass through student network."""
        features_dict = self.student_extractor(x)
        return features_dict["output"]

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step with proper multi-crop processing."""
        images, *_ = batch

        # Generate multi-scale crops
        crops_160, crops_96 = self.generate_multi_crops(images)

        total_loss = 0.0
        loss_count = 0

        # Process 160x160 crops if they exist
        if crops_160 is not None:
            # Teacher forward pass (no grad)
            with torch.no_grad():
                teacher_features = self.teacher_extractor(crops_160)

                # Update vocabulary from teacher features
                if teacher_features["layer4"] is not None:
                    self.update_vocabulary(teacher_features["layer4"], "L")
                    bow_target_L = self.generate_bow_target(
                        teacher_features["layer4"], "L"
                    )
                else:
                    bow_target_L = None

                if self.use_multiscale and teacher_features["layer3"] is not None:
                    self.update_vocabulary(teacher_features["layer3"], "L1")
                    bow_target_L1 = self.generate_bow_target(
                        teacher_features["layer3"], "L1"
                    )
                else:
                    bow_target_L1 = None

            # Student forward pass
            student_features = self.student_extractor(crops_160)

            # Compute losses for layer4 (L)
            if bow_target_L is not None and student_features["layer4"] is not None:
                bow_pred_L = self.predict_bow(student_features["layer4"], "L")
                loss_L = F.kl_div(bow_pred_L.log(), bow_target_L, reduction="batchmean")
                total_loss += loss_L
                loss_count += 1

            # Compute losses for layer3 (L1)
            if bow_target_L1 is not None and student_features["layer3"] is not None:
                bow_pred_L1 = self.predict_bow(student_features["layer3"], "L1")
                loss_L1 = F.kl_div(
                    bow_pred_L1.log(), bow_target_L1, reduction="batchmean"
                )
                total_loss += loss_L1
                loss_count += 1

        # Process 96x96 crops if they exist
        if crops_96 is not None:
            # Teacher forward pass (no grad)
            with torch.no_grad():
                teacher_features = self.teacher_extractor(crops_96)

                # Generate targets (don't update vocab again for same batch)
                if teacher_features["layer4"] is not None:
                    bow_target_L = self.generate_bow_target(
                        teacher_features["layer4"], "L"
                    )
                else:
                    bow_target_L = None

                if self.use_multiscale and teacher_features["layer3"] is not None:
                    bow_target_L1 = self.generate_bow_target(
                        teacher_features["layer3"], "L1"
                    )
                else:
                    bow_target_L1 = None

            # Student forward pass
            student_features = self.student_extractor(crops_96)

            # Compute losses for layer4 (L)
            if bow_target_L is not None and student_features["layer4"] is not None:
                bow_pred_L = self.predict_bow(student_features["layer4"], "L")
                loss_L = F.kl_div(bow_pred_L.log(), bow_target_L, reduction="batchmean")
                total_loss += loss_L
                loss_count += 1

            # Compute losses for layer3 (L1)
            if bow_target_L1 is not None and student_features["layer3"] is not None:
                bow_pred_L1 = self.predict_bow(student_features["layer3"], "L1")
                loss_L1 = F.kl_div(
                    bow_pred_L1.log(), bow_target_L1, reduction="batchmean"
                )
                total_loss += loss_L1
                loss_count += 1

        # Average loss
        if loss_count > 0:
            avg_loss = total_loss / loss_count
        else:
            avg_loss = torch.tensor(0.0, requires_grad=True, device=self.device)

        # Log metrics
        self.log("train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("vocab_msd_L", self.msd_ema_L, on_step=False, on_epoch=True)
        if self.use_multiscale:
            self.log("vocab_msd_L1", self.msd_ema_L1, on_step=False, on_epoch=True)

        return avg_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update teacher after each batch."""
        self.update_teacher()

    def configure_optimizers(self):
        """Configure optimizer with gradient clipping and better scheduling."""
        # Exclude teacher parameters from optimization
        params_to_optimize = []
        for name, param in self.named_parameters():
            if "teacher" not in name and param.requires_grad:
                params_to_optimize.append(param)

        # Use lower learning rate to start
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=self.lr * 0.1,  # Start with lower LR
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        # Simpler scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr * 0.001
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def on_before_optimizer_step(self, optimizer):
        """Clip gradients to prevent explosion."""
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def validation_step(self, batch: Any, batch_idx: int):
        """Validation step with proper multi-crop processing."""
        images, *_ = batch

        # Generate multi-scale crops (fewer crops for validation)
        crops_160, crops_96 = self.generate_multi_crops(images)

        total_loss = 0.0
        loss_count = 0

        # Process 160x160 crops if they exist
        if crops_160 is not None:
            # Teacher forward pass (no grad)
            with torch.no_grad():
                teacher_features = self.teacher_extractor(crops_160)

                # Generate targets (don't update vocabulary during validation)
                if teacher_features["layer4"] is not None:
                    bow_target_L = self.generate_bow_target(
                        teacher_features["layer4"], "L"
                    )
                else:
                    bow_target_L = None

                if self.use_multiscale and teacher_features["layer3"] is not None:
                    bow_target_L1 = self.generate_bow_target(
                        teacher_features["layer3"], "L1"
                    )
                else:
                    bow_target_L1 = None

            # Student forward pass
            student_features = self.student_extractor(crops_160)

            # Compute losses
            if bow_target_L is not None and student_features["layer4"] is not None:
                bow_pred_L = self.predict_bow(student_features["layer4"], "L")
                loss_L = F.kl_div(bow_pred_L.log(), bow_target_L, reduction="batchmean")
                total_loss += loss_L
                loss_count += 1

            if bow_target_L1 is not None and student_features["layer3"] is not None:
                bow_pred_L1 = self.predict_bow(student_features["layer3"], "L1")
                loss_L1 = F.kl_div(
                    bow_pred_L1.log(), bow_target_L1, reduction="batchmean"
                )
                total_loss += loss_L1
                loss_count += 1

        # Process 96x96 crops if they exist
        if crops_96 is not None:
            # Teacher forward pass (no grad)
            with torch.no_grad():
                teacher_features = self.teacher_extractor(crops_96)

                # Generate targets
                if teacher_features["layer4"] is not None:
                    bow_target_L = self.generate_bow_target(
                        teacher_features["layer4"], "L"
                    )
                else:
                    bow_target_L = None

                if self.use_multiscale and teacher_features["layer3"] is not None:
                    bow_target_L1 = self.generate_bow_target(
                        teacher_features["layer3"], "L1"
                    )
                else:
                    bow_target_L1 = None

            # Student forward pass
            student_features = self.student_extractor(crops_96)

            # Compute losses
            if bow_target_L is not None and student_features["layer4"] is not None:
                bow_pred_L = self.predict_bow(student_features["layer4"], "L")
                loss_L = F.kl_div(bow_pred_L.log(), bow_target_L, reduction="batchmean")
                total_loss += loss_L
                loss_count += 1

            if bow_target_L1 is not None and student_features["layer3"] is not None:
                bow_pred_L1 = self.predict_bow(student_features["layer3"], "L1")
                loss_L1 = F.kl_div(
                    bow_pred_L1.log(), bow_target_L1, reduction="batchmean"
                )
                total_loss += loss_L1
                loss_count += 1

        # Average loss
        if loss_count > 0:
            avg_loss = total_loss / loss_count
        else:
            avg_loss = torch.tensor(0.0, requires_grad=True, device=self.device)

        # Log validation metrics
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_vocab_msd_L", self.msd_ema_L, on_step=False, on_epoch=True)
        if self.use_multiscale:
            self.log("val_vocab_msd_L1", self.msd_ema_L1, on_step=False, on_epoch=True)

        return avg_loss

    def test_step(self, batch: Any, batch_idx: int):
        """Test step."""
        images, *_ = batch

        # Generate multi-scale crops
        crops_160, crops_96 = self.generate_multi_crops(images)

        total_loss = 0.0
        loss_count = 0

        # Process 160x160 crops if they exist
        if crops_160 is not None and len(crops_160) > 0:
            # Process each 160x160 crop individually
            for crop in crops_160:
                crop = crop.unsqueeze(0) if crop.dim() == 3 else crop

                # Teacher forward pass (no grad)
                with torch.no_grad():
                    teacher_features = self.teacher_extractor(crop)

                    # Generate targets
                    if teacher_features["layer4"] is not None:
                        bow_target_L = self.generate_bow_target(
                            teacher_features["layer4"], "L"
                        )
                    else:
                        bow_target_L = torch.ones(
                            crop.shape[0], self.vocab_size, device=crop.device
                        )
                        bow_target_L = F.normalize(bow_target_L, p=1, dim=1)

                    if self.use_multiscale and teacher_features["layer3"] is not None:
                        bow_target_L1 = self.generate_bow_target(
                            teacher_features["layer3"], "L1"
                        )
                    else:
                        bow_target_L1 = torch.ones(
                            crop.shape[0], self.vocab_size, device=crop.device
                        )
                        bow_target_L1 = F.normalize(bow_target_L1, p=1, dim=1)

                # Student forward pass
                student_features = self.student_extractor(crop)

                # Compute losses
                if student_features["layer4"] is not None:
                    bow_pred_L = self.predict_bow(student_features["layer4"], "L")
                    loss_L = F.kl_div(
                        bow_pred_L.log(), bow_target_L, reduction="batchmean"
                    )
                    total_loss += loss_L
                    loss_count += 1

                if self.use_multiscale and student_features["layer3"] is not None:
                    bow_pred_L1 = self.predict_bow(student_features["layer3"], "L1")
                    loss_L1 = F.kl_div(
                        bow_pred_L1.log(), bow_target_L1, reduction="batchmean"
                    )
                    total_loss += loss_L1
                    loss_count += 1

        # Process 96x96 crops if they exist
        if crops_96 is not None and len(crops_96) > 0:
            # Process each 96x96 crop individually
            for crop in crops_96:
                crop = crop.unsqueeze(0) if crop.dim() == 3 else crop

                # Teacher forward pass (no grad)
                with torch.no_grad():
                    teacher_features = self.teacher_extractor(crop)

                    # Generate targets
                    if teacher_features["layer4"] is not None:
                        bow_target_L = self.generate_bow_target(
                            teacher_features["layer4"], "L"
                        )
                    else:
                        bow_target_L = torch.ones(
                            crop.shape[0], self.vocab_size, device=crop.device
                        )
                        bow_target_L = F.normalize(bow_target_L, p=1, dim=1)

                    if self.use_multiscale and teacher_features["layer3"] is not None:
                        bow_target_L1 = self.generate_bow_target(
                            teacher_features["layer3"], "L1"
                        )
                    else:
                        bow_target_L1 = torch.ones(
                            crop.shape[0], self.vocab_size, device=crop.device
                        )
                        bow_target_L1 = F.normalize(bow_target_L1, p=1, dim=1)

                # Student forward pass
                student_features = self.student_extractor(crop)

                # Compute losses
                if student_features["layer4"] is not None:
                    bow_pred_L = self.predict_bow(student_features["layer4"], "L")
                    loss_L = F.kl_div(
                        bow_pred_L.log(), bow_target_L, reduction="batchmean"
                    )
                    total_loss += loss_L
                    loss_count += 1

                if self.use_multiscale and student_features["layer3"] is not None:
                    bow_pred_L1 = self.predict_bow(student_features["layer3"], "L1")
                    loss_L1 = F.kl_div(
                        bow_pred_L1.log(), bow_target_L1, reduction="batchmean"
                    )
                    total_loss += loss_L1
                    loss_count += 1

        # Average loss
        if loss_count > 0:
            avg_loss = total_loss / loss_count
        else:
            avg_loss = torch.tensor(0.0, device=self.device)

        # Log test metrics
        self.log("test_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_vocab_msd_L", self.msd_ema_L, on_step=False, on_epoch=True)
        if self.use_multiscale:
            self.log("test_vocab_msd_L1", self.msd_ema_L1, on_step=False, on_epoch=True)

        return avg_loss

    def get_backbone_for_finetuning(self) -> nn.Module:
        """Extract the trained backbone for fine-tuning."""
        return self.student_extractor.backbone
