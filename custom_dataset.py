import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    """
    A custom dataset class for loading ground truth and degraded images.

    Attributes:
        gt_images (list): List of file paths to ground truth images.
        degraded_images (dict): Dictionary mapping each ground truth image to its degraded images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, gt_folder, degraded_folder, transform=None):
        """
        Args:
            gt_folder (str): Directory with all the ground truth images.
            degraded_folder (str): Directory with all the degraded images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.gt_images = self._get_image_paths(gt_folder)
        self.degraded_images = self._get_degraded_image_paths(degraded_folder)
        self.transform = transform
        self.noise_levels = [15, 25, 50]
        self._validate_dataset()

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.gt_images) * len(self.noise_levels)

    def __getitem__(self, idx):
        """
        Generates one sample of data.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (degraded_image, gt_image) where both are tensors.
        """
        noise_idx = idx % len(self.noise_levels)
        img_idx = idx // len(self.noise_levels)

        gt_image_path = self.gt_images[img_idx]
        gt_image = Image.open(gt_image_path).convert('L')  # Convert to grayscale

        noise_level = self.noise_levels[noise_idx]
        degraded_image_name = f"{os.path.splitext(os.path.basename(gt_image_path))[0]}_noise_{noise_level}.png"
        degraded_image_path = os.path.join(os.path.dirname(gt_image_path).replace('resized_ground_truth_images', 'degraded_images'), degraded_image_name)
        degraded_image = Image.open(degraded_image_path).convert('L')  # Convert to grayscale

        if self.transform:
            gt_image = self.transform(gt_image)
            degraded_image = self.transform(degraded_image)

        return degraded_image, gt_image

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

    def _get_degraded_image_paths(self, folder):
        """
        Retrieves the paths of all degraded images in the specified folder.

        Args:
            folder (str): Directory to search for degraded image files.

        Returns:
            dict: Dictionary mapping ground truth image base names to lists of degraded image paths.
        """
        image_extensions = ('png', 'jpg', 'jpeg')
        degraded_paths = {}
        for f in os.listdir(folder):
            if f.lower().endswith(image_extensions):
                base_name = "_".join(f.split('_')[:-2]) + ".png"
                if base_name not in degraded_paths:
                    degraded_paths[base_name] = []
                degraded_paths[base_name].append(os.path.join(folder, f))
        return degraded_paths

    def _validate_dataset(self):
        """
        Validates that the dataset is properly initialized.

        Raises:
            ValueError: If the number of ground truth images does not match the number of degraded images.
        """
        for gt_image in self.gt_images:
            base_name = os.path.basename(gt_image)
            if base_name not in self.degraded_images or len(self.degraded_images[base_name]) != len(self.noise_levels):
                raise ValueError(f"Mismatch or missing degraded images for {base_name}.")
