"""
TorchFreya Dataset Classes
Handles multiple image formats and supports both SSRL and downstream tasks.
"""

import os
import glob
from typing import Optional, Callable, Tuple, List, Union, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torchvision import transforms


class FreyaDataset(Dataset):
    """
    Universal dataset class for TorchFreya framework.
    Supports SSRL pretraining and downstream tasks (classification/segmentation).
    """

    SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".npy"}
    SUPPORTED_ANNOTATION_FORMATS = {".png"}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        task_type: str = "ssrl",  # "ssrl", "classification", "segmentation"
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (224, 224),
        num_classes: Optional[int] = None,
        load_annotations: bool = True,
        cache_images: bool = False,
        valid_extensions: Optional[List[str]] = None,
    ):
        """
        Args:
            root_dir: Root directory containing 'images' and 'annotations' folders
            split: Data split ("train", "val", "test")
            task_type: Type of task ("ssrl", "classification", "segmentation")
            transform: Transform to apply to images
            target_transform: Transform to apply to targets/annotations
            image_size: Target image size for resizing
            num_classes: Number of classes (for classification/segmentation)
            load_annotations: Whether to load annotation files
            cache_images: Whether to cache images in memory (for small datasets)
            valid_extensions: List of valid file extensions to load
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.task_type = task_type
        self.image_size = image_size
        self.num_classes = num_classes
        self.load_annotations = load_annotations and task_type != "ssrl"
        self.cache_images = cache_images

        # Set up paths
        self.images_dir = self.root_dir / "images" / split
        self.annotations_dir = self.root_dir / "annotations" / split

        # Validate directories
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")

        if self.load_annotations and not self.annotations_dir.exists():
            raise ValueError(f"Annotations directory not found: {self.annotations_dir}")

        # Get valid extensions
        self.valid_extensions = valid_extensions or list(self.SUPPORTED_IMAGE_FORMATS)

        # Load file paths
        self.image_paths = self._load_image_paths()
        self.annotation_paths = (
            self._load_annotation_paths() if self.load_annotations else []
        )

        # Validate data consistency
        self._validate_data_consistency()

        # Set up transforms
        self.transform = transform or self._get_default_transform()
        self.target_transform = target_transform or self._get_default_target_transform()

        # Cache for images if requested
        self.image_cache = {} if cache_images else None

        print(f"Loaded {len(self.image_paths)} samples for {task_type} {split} split")

    def _load_image_paths(self) -> List[Path]:
        """Load all valid image file paths."""
        image_paths = []

        for ext in self.valid_extensions:
            pattern = f"*{ext}"
            paths = list(self.images_dir.glob(pattern))
            image_paths.extend(paths)

        # Sort paths to ensure consistent ordering
        image_paths.sort(key=lambda x: x.name)

        if not image_paths:
            raise ValueError(f"No valid images found in {self.images_dir}")

        return image_paths

    def _load_annotation_paths(self) -> List[Path]:
        """Load all annotation file paths."""
        if not self.load_annotations:
            return []

        annotation_paths = []

        # For each image, find corresponding annotation
        for img_path in self.image_paths:
            # Change extension to .png and look in annotations directory
            base_name = img_path.stem
            ann_path = self.annotations_dir / f"{base_name}.png"

            if ann_path.exists():
                annotation_paths.append(ann_path)
            else:
                # Try to find any annotation with same base name
                found = False
                for ext in self.SUPPORTED_ANNOTATION_FORMATS:
                    alt_path = self.annotations_dir / f"{base_name}{ext}"
                    if alt_path.exists():
                        annotation_paths.append(alt_path)
                        found = True
                        break

                if not found:
                    raise ValueError(f"No annotation found for image: {img_path}")

        return annotation_paths

    def _validate_data_consistency(self):
        """Validate that images and annotations are consistent."""
        if self.load_annotations:
            if len(self.image_paths) != len(self.annotation_paths):
                raise ValueError(
                    f"Mismatch between number of images ({len(self.image_paths)}) "
                    f"and annotations ({len(self.annotation_paths)})"
                )

    def _get_default_transform(self) -> Callable:
        """Get default image transform based on task type."""
        if self.task_type == "ssrl":
            # For SSRL, we typically want minimal preprocessing
            # as augmentations are handled by the SSRL model
            return transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            # For downstream tasks
            return transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def _get_default_target_transform(self) -> Callable:
        """Get default target transform based on task type."""
        if self.task_type == "segmentation":
            return transforms.Compose(
                [
                    transforms.Resize(self.image_size, interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                ]
            )
        else:
            return transforms.ToTensor()

    def _load_image(self, path: Path) -> Image.Image:
        """Load image from path, handling different formats."""
        if self.image_cache and str(path) in self.image_cache:
            return self.image_cache[str(path)]

        if path.suffix.lower() == ".npy":
            # Load numpy array and convert to PIL Image
            img_array = np.load(path)
            if img_array.ndim == 2:
                # Grayscale
                img = Image.fromarray(img_array, mode="L").convert("RGB")
            elif img_array.ndim == 3:
                # RGB or similar
                img = Image.fromarray(img_array.astype(np.uint8))
            else:
                raise ValueError(f"Unsupported numpy array shape: {img_array.shape}")
        else:
            # Load with PIL
            img = Image.open(path).convert("RGB")

        if self.image_cache:
            self.image_cache[str(path)] = img

        return img

    def _load_annotation(self, path: Path) -> Image.Image:
        """Load annotation from path."""
        # Annotations are typically PNG with indexed colors
        return Image.open(path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get a single sample."""
        # Load image
        img_path = self.image_paths[idx]
        image = self._load_image(img_path)

        # Apply image transform
        if self.transform:
            image = self.transform(image)

        # For SSRL tasks, return only the image
        if self.task_type == "ssrl":
            return image

        # For supervised tasks, load and return annotation/label
        if self.load_annotations:
            ann_path = self.annotation_paths[idx]
            annotation = self._load_annotation(ann_path)

            if self.target_transform:
                annotation = self.target_transform(annotation)

            return image, annotation
        else:
            # If no annotations available, return dummy target
            if self.task_type == "classification":
                target = torch.tensor(0, dtype=torch.long)  # Dummy class
            else:  # segmentation
                target = torch.zeros(
                    1, *self.image_size, dtype=torch.long
                )  # Dummy mask

            return image, target

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a specific sample."""
        img_path = self.image_paths[idx]
        info = {
            "image_path": str(img_path),
            "index": idx,
            "split": self.split,
            "task_type": self.task_type,
        }

        if self.load_annotations and idx < len(self.annotation_paths):
            info["annotation_path"] = str(self.annotation_paths[idx])

        return info


class FreyaDataModule(L.LightningDataModule):
    """
    Lightning DataModule for TorchFreya framework.
    Handles data loading for all splits and task types.
    """

    def __init__(
        self,
        root_dir: str,
        task_type: str = "ssrl",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (224, 224),
        num_classes: Optional[int] = None,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        cache_images: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.task_type = task_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.num_classes = num_classes
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.target_transform = target_transform
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.cache_images = cache_images

        # Store datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages."""

        if stage == "fit" or stage is None:
            # Training dataset
            self.train_dataset = FreyaDataset(
                root_dir=self.root_dir,
                split="train",
                task_type=self.task_type,
                transform=self.train_transform,
                target_transform=self.target_transform,
                image_size=self.image_size,
                num_classes=self.num_classes,
                cache_images=self.cache_images,
            )

            # Validation dataset
            self.val_dataset = FreyaDataset(
                root_dir=self.root_dir,
                split="val",
                task_type=self.task_type,
                transform=self.val_transform,
                target_transform=self.target_transform,
                image_size=self.image_size,
                num_classes=self.num_classes,
                cache_images=self.cache_images,
            )

        if stage == "test" or stage is None:
            # Test dataset
            self.test_dataset = FreyaDataset(
                root_dir=self.root_dir,
                split="test",
                task_type=self.task_type,
                transform=self.test_transform,
                target_transform=self.target_transform,
                image_size=self.image_size,
                num_classes=self.num_classes,
                cache_images=self.cache_images,
            )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=True,  # Important for SSRL methods
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader (same as test)."""
        return self.test_dataloader()


# Utility functions for creating common datasets
def create_ssrl_datamodule(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    **kwargs,
) -> FreyaDataModule:
    """Create a datamodule for SSRL pretraining."""
    return FreyaDataModule(
        root_dir=root_dir,
        task_type="ssrl",
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        **kwargs,
    )


def create_segmentation_datamodule(
    root_dir: str,
    num_classes: int,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    **kwargs,
) -> FreyaDataModule:
    """Create a datamodule for segmentation tasks."""
    return FreyaDataModule(
        root_dir=root_dir,
        task_type="segmentation",
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        **kwargs,
    )


def create_classification_datamodule(
    root_dir: str,
    num_classes: int,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    **kwargs,
) -> FreyaDataModule:
    """Create a datamodule for classification tasks."""
    return FreyaDataModule(
        root_dir=root_dir,
        task_type="classification",
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        **kwargs,
    )
