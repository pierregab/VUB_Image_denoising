import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import sys
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data
from UNet.UNet_model import UNet  # Assuming UNet is defined in this module

def denormalize(tensor, mean=0.5, std=0.5):
    return tensor * std + mean

def calculate_ssim(X, Y, data_range=1.0):
    return ssim(X, Y, data_range=data_range)

def calculate_psnr(X, Y, data_range=1.0):
    mse = np.mean((X - Y) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 10 * np.log10((data_range ** 2) / mse)
    return psnr_value

def compute_metrics(original, processed):
    original_np = denormalize(original.cpu().numpy().squeeze())
    processed_np = denormalize(processed.cpu().numpy().squeeze())
    
    psnr_value = calculate_psnr(original_np, processed_np, data_range=1.0)
    ssim_value = calculate_ssim(original_np, processed_np, data_range=1.0)
    return psnr_value, ssim_value

def plot_example_images(example_images):
    num_levels = len(example_images)
    fig, axs = plt.subplots(num_levels, 4, figsize=(20, 5 * num_levels))
    
    for i, (sigma, images) in enumerate(example_images.items()):
        gt_image, degraded_image, predicted_image, bm3d_image = images
        
        axs[i, 0].imshow(gt_image, cmap='gray')
        axs[i, 0].set_title(f'Ground Truth (Sigma: {sigma})')
        axs[i, 0].axis('off')
        
        axs[i, 1].imshow(degraded_image, cmap='gray')
        axs[i, 1].set_title('Noisy')
        axs[i, 1].axis('off')
        
        axs[i, 2].imshow(predicted_image, cmap='gray')
        axs[i, 2].set_title('Denoised (Model)')
        axs[i, 2].axis('off')
        
        axs[i, 3].imshow(bm3d_image, cmap='gray')
        axs[i, 3].set_title('Denoised (BM3D)')
        axs[i, 3].axis('off')
    
    plt.show()

def evaluate_model_and_plot(diffusion_model_path, unet_model_path, val_loader, device, include_noise_level=False, use_bm3d=False):
    if use_bm3d:
        import bm3d  # Import bm3d only if use_bm3d is True

    # Load the diffusion model
    diffusion_model = torch.load(diffusion_model_path, map_location=device)
    diffusion_model.to(device)
    diffusion_model.eval()

    # Load the UNet model
    unet_model = UNet().to(device)
    unet_checkpoint = torch.load(unet_model_path, map_location=device)
    if 'model_state_dict' in unet_checkpoint:
        unet_model.load_state_dict(unet_checkpoint['model_state_dict'])
    else:
        unet_model.load_state_dict(unet_checkpoint)
    unet_model.eval()

    metrics = {'noise_level': [], 'psnr_degraded': [], 'psnr_diffusion': [], 'psnr_unet': [], 'psnr_bm3d': [], 'ssim_degraded': [], 'ssim_diffusion': [], 'ssim_unet': [], 'ssim_bm3d': []}
    example_images = {}

    for i, data in enumerate(tqdm(val_loader, desc="Evaluating")):
        if include_noise_level:
            degraded_image, gt_image, noise_level = data
        else:
            degraded_image, gt_image = data
            noise_level = None

        degraded_image = degraded_image.to(device)
        gt_image = gt_image.to(device)

        with torch.no_grad():
            predicted_diffusion = diffusion_model(degraded_image)
            predicted_unet = unet_model(degraded_image)

        for j in range(degraded_image.size(0)):
            psnr_degraded, ssim_degraded = compute_metrics(gt_image[j], degraded_image[j])
            psnr_diffusion, ssim_diffusion = compute_metrics(gt_image[j], predicted_diffusion[j])
            psnr_unet, ssim_unet = compute_metrics(gt_image[j], predicted_unet[j])

            degraded_np = denormalize(degraded_image[j].cpu().numpy().squeeze())
            if use_bm3d:
                bm3d_denoised = bm3d.bm3d(degraded_np, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)
                psnr_bm3d = calculate_psnr(denormalize(gt_image[j].cpu().numpy().squeeze()), bm3d_denoised, data_range=1.0)
                ssim_bm3d = calculate_ssim(denormalize(gt_image[j].cpu().numpy().squeeze()), bm3d_denoised)
            else:
                psnr_bm3d = 0
                ssim_bm3d = 0

            metrics['noise_level'].append(noise_level[j].item() if noise_level is not None else 0)
            metrics['psnr_degraded'].append(psnr_degraded)
            metrics['psnr_diffusion'].append(psnr_diffusion)
            metrics['psnr_unet'].append(psnr_unet)
            metrics['psnr_bm3d'].append(psnr_bm3d)
            metrics['ssim_degraded'].append(ssim_degraded)
            metrics['ssim_diffusion'].append(ssim_diffusion)
            metrics['ssim_unet'].append(ssim_unet)
            metrics['ssim_bm3d'].append(ssim_bm3d)

            sigma_level = int(noise_level[j].item()) if noise_level is not None else 0
            if sigma_level not in example_images:
                example_images[sigma_level] = (gt_image[j].cpu().numpy().squeeze(), degraded_np, denormalize(predicted_diffusion[j].cpu().numpy().squeeze()), bm3d_denoised if use_bm3d else None)

    noise_levels = np.array(metrics['noise_level'])
    psnr_degraded = np.array(metrics['psnr_degraded'])
    psnr_diffusion = np.array(metrics['psnr_diffusion'])
    psnr_unet = np.array(metrics['psnr_unet'])
    psnr_bm3d = np.array(metrics['psnr_bm3d'])
    ssim_degraded = np.array(metrics['ssim_degraded'])
    ssim_diffusion = np.array(metrics['ssim_diffusion'])
    ssim_unet = np.array(metrics['ssim_unet'])
    ssim_bm3d = np.array(metrics['ssim_bm3d'])

    unique_noise_levels = sorted(np.unique(noise_levels))

    avg_psnr_degraded = [np.mean(psnr_degraded[noise_levels == nl]) for nl in unique_noise_levels]
    avg_psnr_diffusion = [np.mean(psnr_diffusion[noise_levels == nl]) for nl in unique_noise_levels]
    avg_psnr_unet = [np.mean(psnr_unet[noise_levels == nl]) for nl in unique_noise_levels]
    avg_psnr_bm3d = [np.mean(psnr_bm3d[noise_levels == nl]) for nl in unique_noise_levels] if use_bm3d else []
    avg_ssim_degraded = [np.mean(ssim_degraded[noise_levels == nl]) for nl in unique_noise_levels]
    avg_ssim_diffusion = [np.mean(ssim_diffusion[noise_levels == nl]) for nl in unique_noise_levels]
    avg_ssim_unet = [np.mean(ssim_unet[noise_levels == nl]) for nl in unique_noise_levels]
    avg_ssim_bm3d = [np.mean(ssim_bm3d[noise_levels == nl]) for nl in unique_noise_levels] if use_bm3d else []

    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    axs[0].plot(unique_noise_levels, avg_psnr_degraded, 'o-', label='Degraded', color='red')
    axs[0].plot(unique_noise_levels, avg_psnr_diffusion, 'o-', label='Diffusion Model', color='green')
    axs[0].plot(unique_noise_levels, avg_psnr_unet, 'o-', label='UNet Model', color='purple')
    if use_bm3d:
        axs[0].plot(unique_noise_levels, avg_psnr_bm3d, 'o-', label='BM3D', color='blue')
    axs[0].set_xlabel('Noise Standard Deviation')
    axs[0].set_ylabel('PSNR')
    axs[0].set_title('PSNR value variation curve')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(unique_noise_levels, avg_ssim_degraded, 'o-', label='Degraded', color='red')
    axs[1].plot(unique_noise_levels, avg_ssim_diffusion, 'o-', label='Diffusion Model', color='green')
    axs[1].plot(unique_noise_levels, avg_ssim_unet, 'o-', label='UNet Model', color='purple')
    if use_bm3d:
        axs[1].plot(unique_noise_levels, avg_ssim_bm3d, 'o-', label='BM3D', color='blue')
    axs[1].set_xlabel('Noise Standard Deviation')
    axs[1].set_ylabel('SSIM')
    axs[1].set_title('SSIM value variation curve')
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()

    if example_images:
        plot_example_images(example_images)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    image_folder = 'DIV2K_valid_HR.nosync'
    train_noise_levels = [10, 20, 30, 40, 50, 60, 70, 80]
    val_noise_levels = [10, 20, 30, 40, 50, 60, 70, 80]

    train_loader, val_loader = load_data(image_folder, batch_size=1, num_workers=8, validation_split=0.5, augment=False, dataset_percentage=0.1, only_validation=False, include_noise_level=True, train_noise_levels=train_noise_levels, val_noise_levels=val_noise_levels)

    evaluate_model_and_plot(diffusion_model_path="checkpoints/diffusion_model_checkpointed_full.pth", unet_model_path="checkpoints/unet_denoising.pth", val_loader=val_loader, device=device, include_noise_level=True, use_bm3d=False)
