import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import sys
from scipy.stats import norm
from tqdm import tqdm

from paper_gan import Generator

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data

def denormalize(tensor, mean=0.5, std=0.5):
    """
    Denormalize a tensor that was normalized using mean and std.
    
    Args:
        tensor (torch.Tensor): Normalized tensor.
        mean (float): Mean used for normalization.
        std (float): Standard deviation used for normalization.
    
    Returns:
        torch.Tensor: Denormalized tensor.
    """
    return tensor * std + mean

def calculate_ssim(X, Y, K1=0.01, K2=0.03, L=1):
    """
    Compute the Structural Similarity Index (SSIM) between two images using the formula provided.
    
    Args:
        X (np.array): Original ground truth image.
        Y (np.array): Processed image to compare.
        K1 (float): Constant for luminance (default: 0.01).
        K2 (float): Constant for contrast (default: 0.03).
        L (float): Dynamic range of the pixel values (default: 1 for normalized images).
    
    Returns:
        float: SSIM value.
    """
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2
    
    mu_X = np.mean(X)
    mu_Y = np.mean(Y)
    sigma_X = np.std(X)
    sigma_Y = np.std(Y)
    sigma_XY = np.mean((X - mu_X) * (Y - mu_Y))
    
    l = (2 * mu_X * mu_Y + C1) / (mu_X ** 2 + mu_Y ** 2 + C1)
    c = (2 * sigma_X * sigma_Y + C2) / (sigma_X ** 2 + sigma_Y ** 2 + C2)
    s = (sigma_XY + C3) / (sigma_X * sigma_Y + C3)
    
    return l * c * s

def calculate_psnr(X, Y, data_range=1.0):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images using the formula provided.
    
    Args:
        X (np.array): Original ground truth image.
        Y (np.array): Processed image to compare.
        data_range (float): The data range of the input images.
    
    Returns:
        float: PSNR value.
    """
    mse = np.mean((X - Y) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 10 * np.log10((data_range ** 2) / mse)
    return psnr_value

def compute_metrics(original, processed):
    """
    Compute PSNR and SSIM between original and processed images.

    Args:
        original (torch.Tensor): Original ground truth image.
        processed (torch.Tensor): Processed image to compare.

    Returns:
        tuple: PSNR and SSIM values.
    """
    original_np = denormalize(original.cpu().numpy().squeeze())
    processed_np = denormalize(processed.cpu().numpy().squeeze())
    
    psnr_value = calculate_psnr(original_np, processed_np, data_range=1.0)  # data_range should match the dynamic range of the images
    ssim_value = calculate_ssim(original_np, processed_np, L=1)  # L should match the dynamic range of the images
    return psnr_value, ssim_value

def evaluate_model_and_plot(model, val_loader, device, model_path="best_denoising_unet_b&w.pth", include_noise_level=False):
    """
    Evaluate a model on the validation set and compute metrics.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        model_path (str): Path to the trained model weights.
        include_noise_level (bool): Whether the DataLoader includes noise level in the samples.

    Returns:
        None
    """
    # Load the trained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Dictionary to store PSNR and SSIM values for different noise levels
    metrics = {'noise_level': [], 'psnr_degraded': [], 'psnr_predicted': [], 'ssim_degraded': [], 'ssim_predicted': []}

    # Use tqdm to add a progress bar
    for i, data in enumerate(tqdm(val_loader, desc="Evaluating")):
        if include_noise_level:
            degraded_image, gt_image, noise_level = data
        else:
            degraded_image, gt_image = data
            noise_level = None

        degraded_image = degraded_image.to(device)
        gt_image = gt_image.to(device)

        with torch.no_grad():
            predicted_image, _ = model(degraded_image)

        for j in range(degraded_image.size(0)):
            psnr_degraded, ssim_degraded = compute_metrics(gt_image[j], degraded_image[j])
            psnr_predicted, ssim_predicted = compute_metrics(gt_image[j], predicted_image[j])

            metrics['noise_level'].append(noise_level[j].item() if noise_level is not None else 0)
            metrics['psnr_degraded'].append(psnr_degraded)
            metrics['psnr_predicted'].append(psnr_predicted)
            metrics['ssim_degraded'].append(ssim_degraded)
            metrics['ssim_predicted'].append(ssim_predicted)

    # Debug: Print unique noise levels
    print("Unique noise levels in validation set:", np.unique(metrics['noise_level']))

    # Convert metrics to numpy arrays for easier processing
    noise_levels = np.array(metrics['noise_level'])
    psnr_degraded = np.array(metrics['psnr_degraded'])
    psnr_predicted = np.array(metrics['psnr_predicted'])
    ssim_degraded = np.array(metrics['ssim_degraded'])
    ssim_predicted = np.array(metrics['ssim_predicted'])

    # Unique noise levels
    unique_noise_levels = sorted(np.unique(noise_levels))

    # Average PSNR and SSIM for each noise level
    avg_psnr_degraded = [np.mean(psnr_degraded[noise_levels == nl]) for nl in unique_noise_levels]
    avg_psnr_predicted = [np.mean(psnr_predicted[noise_levels == nl]) for nl in unique_noise_levels]
    avg_ssim_degraded = [np.mean(ssim_degraded[noise_levels == nl]) for nl in unique_noise_levels]
    avg_ssim_predicted = [np.mean(ssim_predicted[noise_levels == nl]) for nl in unique_noise_levels]

    # Debug: Print averaged metrics
    print("Average PSNR Degraded:", avg_psnr_degraded)
    print("Average PSNR Predicted:", avg_psnr_predicted)
    print("Average SSIM Degraded:", avg_ssim_degraded)
    print("Average SSIM Predicted:", avg_ssim_predicted)

    # Plot PSNR
    plt.figure(figsize=(10, 6))
    plt.plot(unique_noise_levels, avg_psnr_degraded, 'o-', label='Degraded', color='red')
    plt.plot(unique_noise_levels, avg_psnr_predicted, 'o-', label='Predicted', color='green')
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('PSNR')
    plt.title('PSNR value variation curve')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot SSIM
    plt.figure(figsize=(10, 6))
    plt.plot(unique_noise_levels, avg_ssim_degraded, 'o-', label='Degraded', color='red')
    plt.plot(unique_noise_levels, avg_ssim_predicted, 'o-', label='Predicted', color='green')
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('SSIM')
    plt.title('SSIM value variation curve')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # Load the dataset using the provided load_data function
    image_folder = 'DIV2K_valid_HR.nosync'
    train_noise_levels = [10, 20, 30, 40, 50, 60, 70, 80]
    val_noise_levels = [10, 20, 30, 40, 50, 60, 70, 80]

    train_loader, val_loader = load_data(image_folder, batch_size=1, num_workers=8, validation_split=0.2, augment=False, dataset_percentage=0.1, only_validation=False, include_noise_level=True, train_noise_levels=train_noise_levels, val_noise_levels=val_noise_levels)

    # Instantiate the model
    in_channels = 1
    out_channels = 1
    conv_block_channels = [32, 16, 8, 4]
    generator = Generator(in_channels, out_channels, conv_block_channels).to(device)

    # Evaluate the model and plot results
    evaluate_model_and_plot(generator, val_loader, device, model_path="runs/paper_gan/generator_epoch_20.pth", include_noise_level=True)
