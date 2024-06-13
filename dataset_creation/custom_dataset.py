import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    """
    A custom dataset class for loading high-resolution images, extracting non-overlapping patches, and generating noisy versions.

    Attributes:
        patch_pairs (list): List of tuples containing file paths and coordinates for each patch.
        transform (callable, optional): Optional transform to be applied on a sample.
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

        Args:
            folder (str): Directory to search for image files.

        Returns:
            list: Sorted list of image file paths.
        """
        image_extensions = ('png', 'jpg', 'jpeg')
        image_paths = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
        )
        return image_paths

    def _extract_patches(self):
        """
        Extracts all non-overlapping 256x256 patches from each image.

        Returns:
            list: List of tuples containing image path and patch coordinates.
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

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (degraded_image, gt_image) where both are tensors.
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
            gt_patch = self.transform(gt_patch)
            noisy_patch = self.transform(noisy_patch)

        return noisy_patch, gt_patch
