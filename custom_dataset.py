import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    """
    A custom dataset class for loading ground truth and degraded images.

    Attributes:
        gt_images (list): List of file paths to ground truth images.
        degraded_images (list): List of file paths to degraded images.
        transform (callable, optional): Optional transform to be applied
                                        on a sample.
    """

    def __init__(self, gt_folder, degraded_folder, transform=None):
        """
        Args:
            gt_folder (str): Directory with all the ground truth images.
            degraded_folder (str): Directory with all the degraded images.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.gt_images = self._get_image_paths(gt_folder)
        self.degraded_images = self._get_image_paths(degraded_folder)
        self.transform = transform
        self._validate_dataset()

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.gt_images)

    def __getitem__(self, idx):
        """
        Generates one sample of data.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (degraded_image, gt_image) where both are tensors.
        """
        gt_image = Image.open(self.gt_images[idx]).convert('L')  # Convert to grayscale
        degraded_image = Image.open(self.degraded_images[idx]).convert('L')  # Convert to grayscale

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

    def _validate_dataset(self):
        """
        Validates that the dataset is properly initialized.

        Raises:
            ValueError: If the number of ground truth and degraded images do not match.
        """
        if len(self.gt_images) != len(self.degraded_images):
            raise ValueError("The number of ground truth images does not match the number of degraded images.")
