import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

class CustomSIDD_Dataset(Dataset):
    """
    A custom dataset class for loading high-resolution images from the SIDD dataset,
    extracting non-overlapping patches, and returning noisy and ground-truth pairs.
    """

    def __init__(self, root_folder, transform=None, use_rgb=False):
        """
        Args:
            root_folder (str): Directory with the Scene_Instances.txt and Data directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            use_rgb (bool): Whether to use RGB images or grayscale.
        """
        print("Initializing dataset...")
        self.data_folder = os.path.join(root_folder, 'Data')
        self.image_pairs = self._get_image_pairs(root_folder)
        self.transform = transform
        self.patch_pairs = self._extract_patches()
        self.use_rgb = use_rgb
        print(f"Dataset initialized with {len(self.patch_pairs)} patches.")

    def _get_image_pairs(self, root_folder):
        """
        Retrieves the pairs of noisy and ground-truth images in the specified folder.
        """
        scene_file = os.path.join(root_folder, 'Scene_Instances.txt')
        image_pairs = []
        print(f"Reading scene instances from {scene_file}...")
        with open(scene_file, 'r') as file:
            scenes = file.read().splitlines()
        
        for scene in scenes:
            dir_path = os.path.join(self.data_folder, scene)
            if os.path.isdir(dir_path):
                noisy_images = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if 'NOISY' in f])
                gt_images = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if 'GT' in f])
                for noisy_img, gt_img in zip(noisy_images, gt_images):
                    image_pairs.append((noisy_img, gt_img))
        print(f"Found {len(image_pairs)} image pairs.")
        return image_pairs

    def _extract_patches(self):
        """
        Extracts all non-overlapping 256x256 patches from each pair of images.
        """
        patch_pairs = []
        patch_size = 256
        print("Extracting patches...")
        for noisy_path, gt_path in self.image_pairs:
            noisy_image = Image.open(noisy_path)
            gt_image = Image.open(gt_path)
            width, height = noisy_image.size

            for top in range(0, height, patch_size):
                for left in range(0, width, patch_size):
                    if top + patch_size <= height and left + patch_size <= width:
                        patch_pairs.append((noisy_path, gt_path, top, left, patch_size, patch_size))
        print(f"Extracted {len(patch_pairs)} patches.")
        return patch_pairs

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.patch_pairs)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        noisy_path, gt_path, top, left, patch_width, patch_height = self.patch_pairs[idx]
        noisy_image = Image.open(noisy_path)
        gt_image = Image.open(gt_path)
        
        if not self.use_rgb:
            noisy_image = noisy_image.convert('L')  # Convert to grayscale if not using RGB
            gt_image = gt_image.convert('L')

        noisy_patch = transforms.functional.crop(noisy_image, top, left, patch_height, patch_width)
        gt_patch = transforms.functional.crop(gt_image, top, left, patch_height, patch_width)

        if self.transform:
            # Ensure the same random seed for both transformations
            seed = random.randint(0, 2**32)
            random.seed(seed)
            noisy_patch = self.transform(noisy_patch)
            random.seed(seed)
            gt_patch = self.transform(gt_patch)

        return noisy_patch, gt_patch

def load_data(root_folder, batch_size=4, num_workers=2, validation_split=0.2, augment=False, dataset_percentage=1.0, only_validation=False, use_rgb=False):
    """
    Load and preprocess the dataset, returning training and validation DataLoaders.

    Parameters:
    - root_folder (str): Path to the folder containing the Scene_Instance.txt and Data directory.
    - batch_size (int): Number of samples per batch.
    - num_workers (int): Number of worker processes for data loading.
    - validation_split (float): Fraction of the dataset to use for validation.
    - augment (bool): Whether to apply data augmentation to the training set.
    - dataset_percentage (float): Percentage of the total dataset to use (0.0 < dataset_percentage <= 1.0).
    - only_validation (bool): If True, load only validation data without splitting.
    - use_rgb (bool): Whether to use RGB images or grayscale.

    Returns:
    - train_loader (DataLoader): DataLoader for the training dataset (None if only_validation is True).
    - val_loader (DataLoader): DataLoader for the validation dataset.
    """

    print("Loading data...")
    # Define basic transform
    if use_rgb:
        normalize_mean, normalize_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]  # RGB normalization
    else:
        normalize_mean, normalize_std = [0.5], [0.5]  # Grayscale normalization

    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    # Define augmentation transform
    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    # Choose transform based on the augment parameter
    transform = augmentation_transform if augment else basic_transform

    if only_validation:
        val_dataset = CustomSIDD_Dataset(root_folder, transform=transform, use_rgb=use_rgb)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        print("Loaded validation data only.")
        return None, val_loader

    dataset = CustomSIDD_Dataset(root_folder, transform=transform, use_rgb=use_rgb)

    # Determine the size of the dataset to use
    total_size = len(dataset)
    subset_size = int(total_size * dataset_percentage)
    print(f"Using {subset_size} out of {total_size} samples.")

    if subset_size < total_size:
        dataset, _ = random_split(dataset, [subset_size, total_size - subset_size])

    # Split dataset into train and validation
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    print(f"Training size: {train_size}, Validation size: {val_size}")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print("Data loaders created.")

    return train_loader, val_loader

def denormalize(tensor, mean, std):
    """
    Denormalizes a tensor that was normalized with the given mean and std.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def plot_examples(data_loader, num_examples=4, use_rgb=True):
    """
    Plot examples of degraded and ground truth images.

    Parameters:
    - data_loader (DataLoader): DataLoader to load the data from.
    - num_examples (int): Number of examples to plot.
    - use_rgb (bool): Whether the images are RGB or grayscale.
    """
    fig, axs = plt.subplots(num_examples, 2, figsize=(10, 5 * num_examples))
    
    mean = [0.5, 0.5, 0.5] if use_rgb else [0.5]
    std = [0.5, 0.5, 0.5] if use_rgb else [0.5]

    example_count = 0
    for batch_num, batch in enumerate(data_loader, 1):
        print(f"Processing batch {batch_num}...")
        degraded_image, gt_image = batch

        for i in range(degraded_image.size(0)):
            if example_count >= num_examples:
                break

            degraded_np = denormalize(degraded_image[i].cpu(), mean, std).numpy()
            gt_np = denormalize(gt_image[i].cpu(), mean, std).numpy()

            if use_rgb:
                degraded_np = np.transpose(degraded_np, (1, 2, 0))
                gt_np = np.transpose(gt_np, (1, 2, 0))
                cmap = None
            else:
                degraded_np = degraded_np.squeeze()
                gt_np = gt_np.squeeze()
                cmap = 'gray'

            axs[example_count, 0].imshow(degraded_np, cmap=cmap)
            axs[example_count, 0].set_title('Noisy Image')
            axs[example_count, 0].axis('off')

            axs[example_count, 1].imshow(gt_np, cmap=cmap)
            axs[example_count, 1].set_title('Ground Truth Image')
            axs[example_count, 1].axis('off')

            print(f"Displayed example {example_count + 1}")

            example_count += 1

            if example_count >= num_examples:
                break

        if example_count >= num_examples:
            break

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    root_folder = '/Users/pierregabrielbibalsobeaux/Documents/python/VUB_git/VUB_Image_denoising/SIDD_dataset.nosync/SIDD_Medium_Srgb/'
    batch_size = 4
    num_workers = 2  # Reduced number of workers
    validation_split = 0.2
    augment = False
    dataset_percentage = 1.0
    use_rgb = True

    train_loader, val_loader = load_data(
        root_folder=root_folder,
        batch_size=batch_size,
        num_workers=num_workers,
        validation_split=validation_split,
        augment=augment,
        dataset_percentage=dataset_percentage,
        use_rgb=use_rgb
    )

    # Plot examples from the validation set
    plot_examples(val_loader, use_rgb=use_rgb)
