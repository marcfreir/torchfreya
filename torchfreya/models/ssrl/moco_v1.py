from typing import Any, Callable, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.functional import normalize
import lightning as L
from torchvision import transforms
from PIL import Image
from torchfreya.augmentations.symbiosis_augmentations_v1 import RandomAugmentations
import copy


class ProjectionHead(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        """A simple projection head for contrastive learning.

        This class projects input features to a lower-dimensional space using a
        linear layer, typically used in the MoCo framework to generate embeddings
        for contrastive loss computation.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input features.
        output_dim : int
            Dimensionality of the output projections.
        """
        super(ProjectionHead, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the projection head.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Projected features, shape (batch_size, output_dim).
        """
        return self.layer(x)


class MoCo(L.LightningModule):

    def __init__(
        self,
        backbone: nn.Module,
        projector_dim: int,
        output_dim: int = 128,
        K: int = 65536,
        m: float = 0.999,
        temperature: float = 0.07,
        lr: float = 1e-3,
        test_metric: Optional[Callable] = None,
        num_classes: Optional[int] = None,
        augmentations: Optional[Callable] = None,
    ):
        """Momentum Contrast (MoCo) model for self-supervised learning.

        Implements the MoCo framework as described in He et al. (2020). It uses a
        query encoder and a momentum-updated key encoder with a queue to perform
        contrastive learning. Augmentations can be customized via a parameter.

        Parameters
        ----------
        backbone : nn.Module
            Backbone network (e.g., ResNet) for feature extraction.
        projector_dim : int
            Dimensionality of features after the backbone, before projection.
        output_dim : int, optional
            Dimensionality of the output projections, by default 128.
        K : int, optional
            Size of the queue (dictionary), by default 65536.
        m : float, optional
            Momentum coefficient for key encoder update, by default 0.999.
        temperature : float, optional
            Temperature parameter for InfoNCE loss, by default 0.07.
        lr : float, optional
            Learning rate for the optimizer, by default 1e-3.
        test_metric : Optional[Callable], optional
            Metric function for evaluation, unused in this implementation.
        num_classes : Optional[int], optional
            Number of classes, unused in this implementation.
        augmentations : Optional[Callable], optional
            Callable that takes a PIL Image and returns a tuple where the first
            element is an augmented tensor. Defaults to
            RandomAugmentations(apply_masking=False) if None.
        """
        super(MoCo, self).__init__()

        # Query encoder (trainable)
        self.encoder_q = backbone
        self.avgpool_q = nn.AdaptiveAvgPool2d((1, 1))
        self.projector_q = ProjectionHead(projector_dim, output_dim)

        # Key encoder (momentum updated)
        self.encoder_k = copy.deepcopy(backbone)
        self.avgpool_k = nn.AdaptiveAvgPool2d((1, 1))
        self.projector_k = copy.deepcopy(self.projector_q)

        # Disable gradients for key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False

        self.K = K
        self.m = m
        self.temperature = temperature
        self.lr = lr
        self.test_metric = test_metric
        self.num_classes = num_classes

        # Set augmentations, defaulting if not provided
        self.augmentations = (
            RandomAugmentations(apply_masking=False)
            if augmentations is None
            else augmentations
        )
        self.to_pil = transforms.ToPILImage()

        # Initialize queue
        self.register_buffer("queue", torch.randn(output_dim, K))
        self.queue = normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Initialize key encoder weights
        self._init_key_encoder()

    def _init_key_encoder(self):
        """Initialize key encoder with query encoder parameters.

        Copies the weights from the query encoder to the key encoder to ensure
        they start with identical parameters.
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Update key encoder using momentum.

        Updates the key encoder's parameters as a moving average of the query
        encoder's parameters, controlled by the momentum coefficient.
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update queue by enqueuing new keys and dequeuing old ones.

        Maintains a fixed-size queue of keys for contrastive learning.

        Parameters
        ----------
        keys : torch.Tensor
            New keys to enqueue, shape (batch_size, output_dim).
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, (
            f"Queue size ({self.K}) must be divisible by " f"batch size ({batch_size})"
        )

        # Replace keys at pointer position
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shuffle batch for distributed training (Shuffling BN).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, channels, height, width).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Shuffled tensor and indices to unshuffle.
        """
        batch_size = x.size(0)
        idx_shuffle = torch.randperm(batch_size).to(x.device)
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(
        self, x: torch.Tensor, idx_unshuffle: torch.Tensor
    ) -> torch.Tensor:
        """Undo batch shuffle for distributed training.

        Parameters
        ----------
        x : torch.Tensor
            Shuffled tensor.
        idx_unshuffle : torch.Tensor
            Indices to restore original order.

        Returns
        -------
        torch.Tensor
            Unshuffled tensor.
        """
        return x[idx_unshuffle]

    def forward_query(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the query encoder.

        Processes input through the backbone, applies pooling, flattens, projects,
        and normalizes the result.

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Normalized projections, shape (batch_size, output_dim).
        """
        features = self.encoder_q(x)
        pooled = self.avgpool_q(features)
        flattened = torch.flatten(pooled, 1)
        projections = self.projector_q(flattened)
        return normalize(projections, dim=1)

    def forward_key(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the key encoder.

        Similar to forward_query but uses the momentum-updated key encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Normalized projections, shape (batch_size, output_dim).
        """
        features = self.encoder_k(x)
        pooled = self.avgpool_k(features)
        flattened = torch.flatten(pooled, 1)
        projections = self.projector_k(flattened)
        return normalize(projections, dim=1)

    def _augment_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate two augmented views of the input batch.

        Applies the specified augmentations twice per image to create query and
        key views for contrastive learning.

        Parameters
        ----------
        x : torch.Tensor
            Input batch, shape (batch_size, channels, height, width).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Two augmented batches, each shape (batch_size, channels, height, width).
        """
        batch_size = x.size(0)
        aug1, aug2 = [], []

        for i in range(batch_size):
            img_tensor = x[i].cpu()
            pil_img = self.to_pil(img_tensor)
            aug1_img, _ = self.augmentations(pil_img)
            aug2_img, _ = self.augmentations(pil_img)
            aug1.append(aug1_img)
            aug2.append(aug2_img)

        aug1 = torch.stack(aug1).to(self.device)
        aug2 = torch.stack(aug2).to(self.device)
        return aug1, aug2

    def contrastive_loss(
        self, q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss.

        Measures similarity between query and positive key against negative keys
        in the queue, as per the MoCo paper.

        Parameters
        ----------
        q : torch.Tensor
            Query projections, shape (batch_size, output_dim).
        k : torch.Tensor
            Positive key projections, shape (batch_size, output_dim).
        queue : torch.Tensor
            Queue of negative keys, shape (output_dim, K).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, queue])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        return nn.CrossEntropyLoss()(logits, labels)

    def _shared_step(self, batch: Any, prefix: str) -> torch.Tensor:
        """Shared logic for training, validation, and test steps.

        Parameters
        ----------
        batch : Any
            Input batch, typically (data, label) or just data.
        prefix : str
            Prefix for logging (e.g., 'train', 'val', 'test').

        Returns
        -------
        torch.Tensor
            Computed loss for the batch.
        """
        if isinstance(batch, (tuple, list)):
            x, _ = batch
        else:
            x = batch

        x_q, x_k = self._augment_batch(x)
        q = self.forward_query(x_q)
        with torch.no_grad():
            x_k_shuffled, idx_unshuffle = self._batch_shuffle_ddp(x_k)
            k = self.forward_key(x_k_shuffled)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        loss = self.contrastive_loss(q, k, self.queue.clone().detach())
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a training step.

        Computes loss, updates the key encoder, and manages the queue.

        Parameters
        ----------
        batch : Any
            Current batch of data.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Loss value for the batch.
        """
        loss = self._shared_step(batch, "train")

        with torch.no_grad():
            x, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)
            _, x_k = self._augment_batch(x)
            x_k_shuffled, idx_unshuffle = self._batch_shuffle_ddp(x_k)
            k = self.forward_key(x_k_shuffled)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            self._momentum_update_key_encoder()
            self._dequeue_and_enqueue(k)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a validation step.

        Parameters
        ----------
        batch : Any
            Current batch of data.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Loss value for the batch.
        """
        return self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a test step.

        Parameters
        ----------
        batch : Any
            Current batch of data.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Loss value for the batch.
        """
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        """Configure the optimizer.

        Uses SGD with momentum and weight decay as per the MoCo paper.

        Returns
        -------
        torch.optim.Optimizer
            Configured SGD optimizer.
        """
        return torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
