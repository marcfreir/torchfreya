import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from scipy.ndimage import zoom


class RandomAugmentations:
    def __init__(
        self,
        image_size=(240, 240),
        mask_ratio=0.3,
        apply_masking=True,
        layer_aware_crop_prob=0.7,
        scale_range=(0.8, 1.2),
        layer_thickness_range=(5, 20),  # Expected layer thickness in pixels
        preserve_aspect_ratio=True,
    ):
        self.image_size = image_size
        self.mask_ratio = mask_ratio
        self.apply_masking = apply_masking
        self.layer_aware_crop_prob = layer_aware_crop_prob
        self.scale_range = scale_range
        self.layer_thickness_range = layer_thickness_range
        self.preserve_aspect_ratio = preserve_aspect_ratio

        # Minimal standard transform - only ToTensor
        self.to_tensor = transforms.ToTensor()

    def detect_horizontal_layers(self, image_np):
        """
        Detect horizontal geological layers in seismic images using gradient analysis
        """
        if len(image_np.shape) == 3:
            # Convert to grayscale for layer detection
            gray = np.mean(image_np, axis=2)
        else:
            gray = image_np

        # Calculate horizontal gradients to find layer boundaries
        grad_y = np.gradient(gray, axis=0)

        # Find strong horizontal gradients (layer boundaries)
        gradient_strength = np.abs(grad_y)

        # Average gradient strength across width to get horizontal profile
        horizontal_profile = np.mean(gradient_strength, axis=1)

        # Find peaks that represent layer boundaries
        threshold = np.percentile(horizontal_profile, 75)
        layer_boundaries = np.where(horizontal_profile > threshold)[0]

        return layer_boundaries, horizontal_profile

    def layer_aware_crop(self, image):
        """
        Perform cropping that respects geological layer structure
        """
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        # Detect layers
        layer_boundaries, _ = self.detect_horizontal_layers(image_np)

        if len(layer_boundaries) < 2:
            # Fallback to random crop if no clear layers detected
            return self.random_crop(image)

        # Choose crop region that includes complete layers
        min_layer_thickness = self.layer_thickness_range[0]
        max_layer_thickness = self.layer_thickness_range[1]

        # Find a good crop region
        target_h = self.image_size[0]
        target_w = self.image_size[1]

        # Try to find a crop that starts and ends at layer boundaries
        valid_crops = []

        for start_boundary in layer_boundaries:
            for end_boundary in layer_boundaries:
                if end_boundary > start_boundary:
                    crop_height = end_boundary - start_boundary
                    if (
                        abs(crop_height - target_h) < target_h * 0.3
                    ):  # Within 30% of target
                        valid_crops.append((start_boundary, end_boundary))

        if valid_crops:
            # Choose random valid crop
            start_y, end_y = random.choice(valid_crops)
            crop_height = end_y - start_y
        else:
            # Fallback: crop around a layer boundary
            center_y = random.choice(layer_boundaries)
            start_y = max(0, center_y - target_h // 2)
            end_y = min(h, start_y + target_h)
            start_y = max(0, end_y - target_h)
            crop_height = end_y - start_y

        # Random horizontal crop
        if w > target_w:
            start_x = random.randint(0, w - target_w)
            end_x = start_x + target_w
        else:
            start_x, end_x = 0, w

        # Perform crop
        cropped = image_np[start_y:end_y, start_x:end_x]

        # Resize to exact target size if needed
        if cropped.shape[:2] != (target_h, target_w):
            cropped = self.resize_image(cropped, (target_h, target_w))

        return Image.fromarray(cropped.astype(np.uint8))

    def random_crop(self, image):
        """
        Fallback random crop when layer detection fails
        """
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        target_h, target_w = self.image_size

        if h > target_h:
            start_y = random.randint(0, h - target_h)
            end_y = start_y + target_h
        else:
            start_y, end_y = 0, h

        if w > target_w:
            start_x = random.randint(0, w - target_w)
            end_x = start_x + target_w
        else:
            start_x, end_x = 0, w

        cropped = image_np[start_y:end_y, start_x:end_x]

        if cropped.shape[:2] != (target_h, target_w):
            cropped = self.resize_image(cropped, (target_h, target_w))

        return Image.fromarray(cropped.astype(np.uint8))

    def smart_scale(self, image):
        """
        Apply scaling (zoom in/out) while preserving layer structure
        """
        scale_factor = random.uniform(*self.scale_range)
        image_np = np.array(image)

        if len(image_np.shape) == 3:
            h, w, c = image_np.shape
            if self.preserve_aspect_ratio:
                # Scale uniformly
                scaled = zoom(image_np, (scale_factor, scale_factor, 1), order=1)
            else:
                # Allow different scaling for height and width
                scale_y = random.uniform(*self.scale_range)
                scale_x = random.uniform(*self.scale_range)
                scaled = zoom(image_np, (scale_y, scale_x, 1), order=1)
        else:
            h, w = image_np.shape
            if self.preserve_aspect_ratio:
                scaled = zoom(image_np, (scale_factor, scale_factor), order=1)
            else:
                scale_y = random.uniform(*self.scale_range)
                scale_x = random.uniform(*self.scale_range)
                scaled = zoom(image_np, (scale_y, scale_x), order=1)

        # Convert back to PIL Image
        scaled = np.clip(scaled, 0, 255).astype(np.uint8)
        return Image.fromarray(scaled)

    def resize_image(self, image_np, target_size):
        """
        Resize image array to target size
        """
        if len(image_np.shape) == 3:
            h, w, c = image_np.shape
            scale_y = target_size[0] / h
            scale_x = target_size[1] / w
            resized = zoom(image_np, (scale_y, scale_x, 1), order=1)
        else:
            h, w = image_np.shape
            scale_y = target_size[0] / h
            scale_x = target_size[1] / w
            resized = zoom(image_np, (scale_y, scale_x), order=1)

        return resized

    def create_masked_image(self, image):
        """
        Create masked version for self-supervised learning
        """
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        patch_size = (32, 32)

        num_patches_h = h // patch_size[0]
        num_patches_w = w // patch_size[1]
        total_patches = num_patches_h * num_patches_w
        num_masked_patches = int(total_patches * self.mask_ratio)

        patch_indices = np.random.choice(
            total_patches, num_masked_patches, replace=False
        )
        mask = np.ones((h, w), dtype=bool)

        for idx in patch_indices:
            row_idx = idx // num_patches_w
            col_idx = idx % num_patches_w
            start_h = row_idx * patch_size[0]
            end_h = min((row_idx + 1) * patch_size[0], h)
            start_w = col_idx * patch_size[1]
            end_w = min((col_idx + 1) * patch_size[1], w)
            mask[start_h:end_h, start_w:end_w] = 0

        masked_image_np = image_np.copy()
        masked_image_np[~mask] = 0
        masked_image = Image.fromarray(masked_image_np)
        mask_tensor = torch.tensor(~mask, dtype=torch.float32).unsqueeze(0)

        return masked_image, mask_tensor

    def __call__(self, image):
        """
        Apply augmentations to input image
        """
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        # Apply scaling transformation
        if random.random() < 0.6:  # 60% chance to apply scaling
            image = self.smart_scale(image)

        # Apply layer-aware cropping
        if random.random() < self.layer_aware_crop_prob:
            aug_image = self.layer_aware_crop(image)
        else:
            aug_image = self.random_crop(image)

        # Convert to tensor
        aug_image = self.to_tensor(aug_image)

        # Apply masking for self-supervised learning
        mask = None
        if self.apply_masking and random.random() < 0.5:
            # Convert back to PIL for masking
            aug_image_pil = transforms.ToPILImage()(aug_image)
            masked_image, mask = self.create_masked_image(aug_image_pil)
            aug_image = self.to_tensor(masked_image)

        return aug_image, mask
