import random
from scipy.ndimage import gaussian_filter
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class RandomAugmentations:
    def __init__(
        self,
        image_size=(240, 240),
        mask_ratio=0.3,
        apply_masking=True,
        gaussian_noise_std=0.1,
        patch_height=40,
        relief_sigma=1.0,
        relief_intensity=0.5,
    ):
        self.image_size = image_size
        self.mask_ratio = mask_ratio
        self.apply_masking = apply_masking
        self.gaussian_noise_std = gaussian_noise_std
        self.patch_height = patch_height
        self.relief_sigma = relief_sigma
        self.relief_intensity = relief_intensity

        # Standard augmentations
        self.standard_augment = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def create_masked_image(self, image):
        image_np = np.array(image)
        h, w, _ = image_np.shape
        patch_size = (32, 32)  # You could make this configurable too

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
            mask[
                row_idx * patch_size[0] : (row_idx + 1) * patch_size[0],
                col_idx * patch_size[1] : (col_idx + 1) * patch_size[1],
            ] = 0

        masked_image_np = image_np.copy()
        masked_image_np[~mask] = 0
        masked_image = Image.fromarray(masked_image_np)
        mask_tensor = torch.tensor(~mask, dtype=torch.float32).unsqueeze(0)

        return masked_image, mask_tensor

    def add_gaussian_noise(self, image):
        noise = torch.randn(image.size()) * self.gaussian_noise_std
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)

    def vertical_patch_swap(self, image):
        image_np = np.array(image)
        height, width, channels = image_np.shape
        num_patches = height // self.patch_height + (
            1 if height % self.patch_height != 0 else 0
        )

        patches = []
        for i in range(num_patches):
            start_row = i * self.patch_height
            end_row = min((i + 1) * self.patch_height, height)
            patches.append(image_np[start_row:end_row, :, :])

        random.shuffle(patches)
        shuffled_image = np.vstack(patches)
        return Image.fromarray(shuffled_image)

    def pseudo_relief_transform(self, image):
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_gray = (
            np.mean(image_np, axis=-1) if image_np.shape[-1] > 1 else image_np.squeeze()
        )

        grad_x = gaussian_filter(image_gray, sigma=self.relief_sigma, order=[0, 1])
        grad_y = gaussian_filter(image_gray, sigma=self.relief_sigma, order=[1, 0])

        light_angle = random.uniform(0, 2 * np.pi)
        relief = self.relief_intensity * (
            np.cos(light_angle) * grad_x + np.sin(light_angle) * grad_y
        )

        image_relief = image_gray + relief
        image_relief = np.clip(image_relief, 0, 1)
        if image_np.shape[-1] > 1:
            image_relief = np.stack([image_relief] * image_np.shape[-1], axis=-1)

        return torch.tensor(image_relief).permute(2, 0, 1).to(image.device)

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        aug_image = self.standard_augment(image)

        if random.random() < 0.5:
            aug_image = self.add_gaussian_noise(aug_image)
        if random.random() < 0.5:
            aug_image_pil = transforms.ToPILImage()(aug_image)
            aug_image = transforms.ToTensor()(self.vertical_patch_swap(aug_image_pil))
        if random.random() < 1.0:
            aug_image = self.pseudo_relief_transform(aug_image)

        mask = None
        if self.apply_masking and random.random() < 0.5:
            aug_image_np = (aug_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            aug_image_pil = Image.fromarray(aug_image_np)
            masked_image, mask = self.create_masked_image(aug_image_pil)
            aug_image = transforms.ToTensor()(masked_image)

        return aug_image, mask
