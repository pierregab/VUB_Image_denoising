import os
import torch
from torch.utils.data import DataLoader
from dataset_creation.custom_dataset import CustomDataset  # Ensure this is the correct import path

def summarize_dataset(image_folder, patch_size=256, noise_levels=[15, 25, 50], use_rgb=False):
    # Initialize the dataset
    dataset = CustomDataset(image_folder=image_folder, noise_levels=noise_levels, use_rgb=use_rgb)
    
    total_images = len(dataset.image_paths)
    total_patches = len(dataset.patch_pairs)
    patches_per_image = total_patches / total_images
    total_samples = len(dataset)
    
    # Print dataset summary
    print(f"Total number of images: {total_images}")
    print(f"Total number of patches: {total_patches}")
    print(f"Average patches per image: {patches_per_image:.2f}")
    print(f"Total number of samples (patches x noise levels): {total_samples}")
    print(f"Noise levels: {noise_levels}")
    
    # For table purposes, create a dictionary of stats
    summary = {
        "Total Images": total_images,
        "Total Patches": total_patches,
        "Average Patches per Image": round(patches_per_image, 2),
        "Total Samples": total_samples,
        "Noise Levels": noise_levels
    }
    
    return summary

# Example usage
image_folder = 'path/to/DIV2K_train_HR'  # Replace with the actual path
summary = summarize_dataset(image_folder)

# Optional: print the summary in a table-like format
print("\nDataset Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")
