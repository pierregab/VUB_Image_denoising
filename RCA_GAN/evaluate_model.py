import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import sys
from tqdm import tqdm
import bm3d

from paper_gan import Generator

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data

def denormalize(tensor, mean=0.5, std=0.5):
    return tensor * std + mean

def calculate_ssim(X, Y, K1=0.01, K2=0.03, L=1):
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
    mse = np.mean((X - Y) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 10 * np.log10((data_range ** 2) / mse)
    return psnr_value

def compute_metrics(original, processed):
    original_np = denormalize(original.cpu().numpy().squeeze())
    processed_np = denormalize(processed.cpu().numpy().squeeze())
    
    psnr_value = calculate_psnr(original_np, processed_np, data_range=1.0)
    ssim_value = calculate_ssim(original_np, processed_np, L=1)
    return psnr_value, ssim_value

def bm3d_denoise(image, sigma):
    return bm3d.bm3d(image, sigma)

def plot_example_image(gt_image, degraded_image, predicted_image, bm3d_image):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    axs[0].imshow(gt_image, cmap='gray')
    axs[0].set_title('Ground Truth')
    axs[0].axis('off')
    
    axs[1].imshow(degraded_image, cmap='gray')
    axs[1].set_title('Noisy')
    axs[1].axis('off')
    
    axs[2].imshow(predicted_image, cmap='gray')
    axs[2].set_title('Denoised (Model)')
    axs[2].axis('off')
    
    axs[3].imshow(bm3d_image, cmap='gray')
    axs[3].set_title('Denoised (BM3D)')
    axs[3].axis('off')
    
    plt.show()

def evaluate_model_and_plot(model, val_loader, device, model_path="best_denoising_unet_b&w.pth", include_noise_level=False):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    metrics = {'noise_level': [], 'psnr_degraded': [], 'psnr_predicted': [], 'psnr_bm3d': [], 'ssim_degraded': [], 'ssim_predicted': [], 'ssim_bm3d': []}

    example_images = None

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

            degraded_np = denormalize(degraded_image[j].cpu().numpy().squeeze())
            sigma = noise_level[j].item() / 255.0 if noise_level is not None else 0
            bm3d_denoised = bm3d_denoise(degraded_np, sigma)
            psnr_bm3d = calculate_psnr(denormalize(gt_image[j].cpu().numpy().squeeze()), bm3d_denoised, data_range=1.0)
            ssim_bm3d = calculate_ssim(denormalize(gt_image[j].cpu().numpy().squeeze()), bm3d_denoised, L=1)

            metrics['noise_level'].append(noise_level[j].item() if noise_level is not None else 0)
            metrics['psnr_degraded'].append(psnr_degraded)
            metrics['psnr_predicted'].append(psnr_predicted)
            metrics['psnr_bm3d'].append(psnr_bm3d)
            metrics['ssim_degraded'].append(ssim_degraded)
            metrics['ssim_predicted'].append(ssim_predicted)
            metrics['ssim_bm3d'].append(ssim_bm3d)

            if example_images is None:
                example_images = (gt_image[j].cpu().numpy().squeeze(), degraded_np, denormalize(predicted_image[j].cpu().numpy().squeeze()), bm3d_denoised)

    noise_levels = np.array(metrics['noise_level'])
    psnr_degraded = np.array(metrics['psnr_degraded'])
    psnr_predicted = np.array(metrics['psnr_predicted'])
    psnr_bm3d = np.array(metrics['psnr_bm3d'])
    ssim_degraded = np.array(metrics['ssim_degraded'])
    ssim_predicted = np.array(metrics['ssim_predicted'])
    ssim_bm3d = np.array(metrics['ssim_bm3d'])

    unique_noise_levels = sorted(np.unique(noise_levels))

    avg_psnr_degraded = [np.mean(psnr_degraded[noise_levels == nl]) for nl in unique_noise_levels]
    avg_psnr_predicted = [np.mean(psnr_predicted[noise_levels == nl]) for nl in unique_noise_levels]
    avg_psnr_bm3d = [np.mean(psnr_bm3d[noise_levels == nl]) for nl in unique_noise_levels]
    avg_ssim_degraded = [np.mean(ssim_degraded[noise_levels == nl]) for nl in unique_noise_levels]
    avg_ssim_predicted = [np.mean(ssim_predicted[noise_levels == nl]) for nl in unique_noise_levels]
    avg_ssim_bm3d = [np.mean(ssim_bm3d[noise_levels == nl]) for nl in unique_noise_levels]

    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    axs[0].plot(unique_noise_levels, avg_psnr_degraded, 'o-', label='Degraded', color='red')
    axs[0].plot(unique_noise_levels, avg_psnr_predicted, 'o-', label='Predicted', color='green')
    axs[0].plot(unique_noise_levels, avg_psnr_bm3d, 'o-', label='BM3D', color='blue')
    axs[0].set_xlabel('Noise Standard Deviation')
    axs[0].set_ylabel('PSNR')
    axs[0].set_title('PSNR value variation curve')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(unique_noise_levels, avg_ssim_degraded, 'o-', label='Degraded', color='red')
    axs[1].plot(unique_noise_levels, avg_ssim_predicted, 'o-', label='Predicted', color='green')
    axs[1].plot(unique_noise_levels, avg_ssim_bm3d, 'o-', label='BM3D', color='blue')
    axs[1].set_xlabel('Noise Standard Deviation')
    axs[1].set_ylabel('SSIM')
    axs[1].set_title('SSIM value variation curve')
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()

    if example_images is not None:
        plot_example_image(*example_images)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    image_folder = 'DIV2K_valid_HR.nosync'
    train_noise_levels = [10, 20, 30, 40, 50, 60, 70, 80]
    val_noise_levels = [10, 20, 30, 40, 50, 60, 70, 80]

    train_loader, val_loader = load_data(image_folder, batch_size=1, num_workers=8, validation_split=0.2, augment=False, dataset_percentage=0.01, only_validation=False, include_noise_level=True, train_noise_levels=train_noise_levels, val_noise_levels=val_noise_levels)

    in_channels = 1
    out_channels = 1
    conv_block_channels = [32, 16, 8, 4]
    generator = Generator(in_channels, out_channels, conv_block_channels).to(device)

    evaluate_model_and_plot(generator, val_loader, device, model_path="runs/paper_gan/generator_epoch_20.pth", include_noise_level=True)
