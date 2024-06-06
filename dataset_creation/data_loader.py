import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from dataset_creation.custom_dataset import CustomDataset  # Ensure custom_dataset.py is in the same directory or in your PYTHONPATH

def load_data(gt_folder, degraded_folder, batch_size=4, num_workers=4, validation_split=0.2, augment=False, dataset_percentage=1.0):
    """
    Load and preprocess the dataset, returning training and validation DataLoaders.

    Parameters:
    - gt_folder (str): Path to the folder containing ground truth images.
    - degraded_folder (str): Path to the folder containing degraded images.
    - batch_size (int): Number of samples per batch.
    - num_workers (int): Number of worker processes for data loading.
    - validation_split (float): Fraction of the dataset to use for validation.
    - augment (bool): Whether to apply data augmentation to the training set.
    - dataset_percentage (float): Percentage of the total dataset to use (0.0 < dataset_percentage <= 1.0).

    Returns:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    """

    # Define basic transform
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
    ])

    # Define augmentation transform
    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
    ])

    # Choose transform based on the augment parameter
    transform = augmentation_transform if augment else basic_transform

    dataset = CustomDataset(gt_folder, degraded_folder, transform=transform)

    # Determine the size of the dataset to use
    total_size = len(dataset)
    subset_size = int(total_size * dataset_percentage)

    if subset_size < total_size:
        dataset, _ = random_split(dataset, [subset_size, total_size - subset_size])

    # Split dataset into train and validation
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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
    for degraded_image, gt_image in data_loader:
        for i in range(degraded_image.size(0)):
            if example_count >= num_examples:
                break

            degraded_np = degraded_image[i].cpu().squeeze().numpy()
            gt_np = gt_image[i].cpu().squeeze().numpy()

            axs[example_count, 0].imshow(degraded_np, cmap='gray')
            axs[example_count, 0].set_title('Degraded Image')
            axs[example_count, 0].axis('off')

            axs[example_count, 1].imshow(gt_np, cmap='gray')
            axs[example_count, 1].set_title('Ground Truth Image')
            axs[example_count, 1].axis('off')

            example_count += 1

            if example_count >= num_examples:
                break

    plt.tight_layout()
    plt.show()
