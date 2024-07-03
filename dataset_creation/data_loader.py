import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from dataset_creation.custom_dataset import CustomDataset  # Ensure custom_dataset.py is in the same directory or in your PYTHONPATH

def load_data(image_folder, batch_size=4, num_workers=4, validation_split=0.2, augment=False, dataset_percentage=1.0, only_validation=False, include_noise_level=False, train_noise_levels=None, val_noise_levels=None, use_rgb=False):
    """
    Load and preprocess the dataset, returning training and validation DataLoaders.

    Parameters:
    - image_folder (str): Path to the folder containing images.
    - batch_size (int): Number of samples per batch.
    - num_workers (int): Number of worker processes for data loading.
    - validation_split (float): Fraction of the dataset to use for validation.
    - augment (bool): Whether to apply data augmentation to the training set.
    - dataset_percentage (float): Percentage of the total dataset to use (0.0 < dataset_percentage <= 1.0).
    - only_validation (bool): If True, load only validation data without splitting.
    - include_noise_level (bool): Whether to include noise level in the returned samples.
    - train_noise_levels (list): List of noise levels to be used for training.
    - val_noise_levels (list): List of noise levels to be used for validation.
    - use_rgb (bool): Whether to use RGB images or grayscale.

    Returns:
    - train_loader (DataLoader): DataLoader for the training dataset (None if only_validation is True).
    - val_loader (DataLoader): DataLoader for the validation dataset.
    """

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
        val_dataset = CustomDataset(image_folder, transform=transform, include_noise_level=include_noise_level, noise_levels=val_noise_levels, use_rgb=use_rgb)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return None, val_loader

    train_dataset = CustomDataset(image_folder, transform=transform, include_noise_level=include_noise_level, noise_levels=train_noise_levels, use_rgb=use_rgb)
    val_dataset = CustomDataset(image_folder, transform=transform, include_noise_level=include_noise_level, noise_levels=val_noise_levels, use_rgb=use_rgb)

    # Debug: Print noise levels used in datasets
    print("Train noise levels:", train_noise_levels)
    print("Validation noise levels:", val_noise_levels)

    # Determine the size of the dataset to use
    total_size = len(train_dataset)
    subset_size = int(total_size * dataset_percentage)

    if subset_size < total_size:
        train_dataset, _ = random_split(train_dataset, [subset_size, total_size - subset_size])

    # Split dataset into train and validation
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def plot_examples(data_loader, num_examples=4):
    """
    Plot examples of degraded and ground truth images.

    Parameters:
    - data_loader (DataLoader): DataLoader to load the data from.
    - num_examples (int): Number of examples to plot.
    """
    fig, axs = plt.subplots(num_examples, 2, figsize=(10, 5 * num_examples))
    
    example_count = 0
    for batch in data_loader:
        if len(batch) == 3:
            degraded_image, gt_image, noise_level = batch
        else:
            degraded_image, gt_image = batch

        for i in range(degraded_image.size(0)):
            if example_count >= num_examples:
                break

            if degraded_image.size(1) == 1:  # Grayscale
                degraded_np = degraded_image[i].cpu().squeeze().numpy()
                gt_np = gt_image[i].cpu().squeeze().numpy()
                cmap = 'gray'
            else:  # RGB
                degraded_np = degraded_image[i].cpu().permute(1, 2, 0).numpy()
                gt_np = gt_image[i].cpu().permute(1, 2, 0).numpy()
                cmap = None

            axs[example_count, 0].imshow(degraded_np, cmap=cmap)
            axs[example_count, 0].set_title('Degraded Image')
            axs[example_count, 0].axis('off')

            axs[example_count, 1].imshow(gt_np, cmap=cmap)
            axs[example_count, 1].set_title('Ground Truth Image')
            axs[example_count, 1].axis('off')

            example_count += 1

            if example_count >= num_examples:
                break

    plt.tight_layout()
    plt.show()

