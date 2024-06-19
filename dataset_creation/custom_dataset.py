import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    """
    A custom dataset class for loading high-resolution images, extracting non-overlapping patches, and generating noisy versions.
    """

    def __init__(self, image_folder, transform=None):
        """
        Args:
            image_folder (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = self._get_image_paths(image_folder)
        self.transform = transform
        self.noise_levels = [15, 25, 50]
        self.patch_pairs = self._extract_patches()

    def _get_image_paths(self, folder):
        """
        Retrieves the paths of all images in the specified folder.
        """
        image_extensions = ('png', 'jpg', 'jpeg')
        image_paths = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
        )
        return image_paths

    def _extract_patches(self):
        """
        Extracts all non-overlapping 256x256 patches from each image.
        """
        patch_pairs = []
        patch_size = 256
        for image_path in self.image_paths:
            image = Image.open(image_path)
            width, height = image.size

            for top in range(0, height, patch_size):
                for left in range(0, width, patch_size):
                    if top + patch_size <= height and left + patch_size <= width:
                        patch_pairs.append((image_path, top, left, patch_size, patch_size))
        return patch_pairs

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.patch_pairs) * len(self.noise_levels)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        noise_idx = idx % len(self.noise_levels)
        patch_idx = idx // len(self.noise_levels)

        image_path, top, left, patch_width, patch_height = self.patch_pairs[patch_idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        gt_patch = transforms.functional.crop(image, top, left, patch_height, patch_width)

        # Apply Gaussian noise to the patch
        noise_level = self.noise_levels[noise_idx]
        noisy_patch = np.array(gt_patch, dtype=np.float32)
        noisy_patch += np.random.normal(scale=noise_level, size=noisy_patch.shape)
        noisy_patch = np.clip(noisy_patch, 0, 255).astype(np.uint8)
        noisy_patch = Image.fromarray(noisy_patch)

        if self.transform:
            # Ensure the same random seed for both transformations
            seed = random.randint(0, 2**32)
            random.seed(seed)
            gt_patch = self.transform(gt_patch)
            random.seed(seed)
            noisy_patch = self.transform(noisy_patch)

        return noisy_patch, gt_patch
