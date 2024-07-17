import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import lpips
from DISTS_pytorch import DISTS
from skimage.metrics import structural_similarity as ssim
import sys
from dataset_creation.data_loader import load_data
from UNet.RDUNet_model import RDUNet
from diffusion_denoising.diffusion_RDUnet import DiffusionModel, RDUNet_T

def denormalize(tensor, mean=0.5, std=0.5):
    return tensor * std + mean

def calculate_ssim(X, Y, data_range=1.0, use_rgb=False):
    if use_rgb:
        return ssim(X, Y, data_range=data_range, multichannel=True, channel_axis=0)
    else:
        return ssim(X, Y, data_range=data_range)

def calculate_psnr(X, Y, data_range=1.0):
    mse = np.mean((X - Y) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 10 * np.log10((data_range ** 2) / mse)
    return psnr_value

def compute_metrics(original, processed, lpips_model, dists_model, use_rgb=True):
    original_np = original.cpu().numpy()
    processed_np = processed.cpu().numpy()

    psnr_value = calculate_psnr(original_np, processed_np, data_range=1.0)
    ssim_value = calculate_ssim(original_np, processed_np, data_range=1.0, use_rgb=use_rgb)

    if use_rgb:
        original_tensor = torch.tensor(original_np).unsqueeze(0).float()
        processed_tensor = torch.tensor(processed_np).unsqueeze(0).float()
    else:
        original_tensor = torch.tensor(original_np).unsqueeze(0).repeat(1, 3, 1, 1).float()
        processed_tensor = torch.tensor(processed_np).unsqueeze(0).repeat(1, 3, 1, 1).float()

    original_tensor = normalize_to_neg1_1(original_tensor)
    processed_tensor = normalize_to_neg1_1(processed_tensor)

    device = next(lpips_model.parameters()).device
    original_tensor = original_tensor.to(device)
    processed_tensor = processed_tensor.to(device)

    with torch.no_grad():
        lpips_value = lpips_model(original_tensor, processed_tensor).item()
        dists_value = dists_model(original_tensor, processed_tensor).item()

    return psnr_value, ssim_value, lpips_value, dists_value

def normalize_to_neg1_1(tensor):
    return 2 * tensor - 1

def evaluate_diffusion_model(epochs, diffusion_model_paths, val_loader, device):
    lpips_model = lpips.LPIPS(net='alex').to(device)
    dists_model = DISTS().to(device)

    best_epoch = None
    best_psnr = -float('inf')
    all_metrics = []

    for epoch, diffusion_model_path in zip(epochs, diffusion_model_paths):
        diffusion_model = DiffusionModel(RDUNet_T(base_filters=64).to(device)).to(device)
        diffusion_checkpoint = torch.load(diffusion_model_path, map_location=device)
        if isinstance(diffusion_checkpoint, dict) and 'model_state_dict' in diffusion_checkpoint:
            diffusion_model.load_state_dict(diffusion_checkpoint['model_state_dict'])
        else:
            diffusion_model = diffusion_checkpoint
        diffusion_model.to(device)
        diffusion_model.eval()

        epoch_psnr = []

        for data in tqdm(val_loader, desc=f"Evaluating Epoch {epoch}"):
            degraded_images, gt_images = data

            degraded_images = degraded_images.to(device)
            gt_images = gt_images.to(device)

            with torch.no_grad():
                predicted_diffusions = diffusion_model.improved_sampling(degraded_images)

            for j in range(degraded_images.size(0)):
                gt_image = gt_images[j]
                predicted_diffusion = predicted_diffusions[j]

                psnr_value, ssim_value, lpips_value, dists_value = compute_metrics(gt_image, predicted_diffusion, lpips_model, dists_model, use_rgb=True)
                epoch_psnr.append(psnr_value)

        avg_psnr = np.mean(epoch_psnr)
        all_metrics.append((epoch, avg_psnr))

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = epoch

    return best_epoch, all_metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    image_folder = 'DIV2K_valid_HR.nosync'
    val_noise_levels = [10, 20, 30, 40, 50, 60, 70, 80]

    _, val_loader = load_data(image_folder, batch_size=1, num_workers=8, validation_split=0.5, augment=False, dataset_percentage=0.01, only_validation=True, include_noise_level=True, val_noise_levels=val_noise_levels, use_rgb=True)

    epochs_to_evaluate = [120, 130, 140, 148, 150]  # List of epochs you want to evaluate
    diffusion_model_paths = [f"checkpoints/diffusion_RDUnet_model_checkpointed_epoch_{epoch}.pth" for epoch in epochs_to_evaluate]

    best_epoch, all_metrics = evaluate_diffusion_model(epochs_to_evaluate, diffusion_model_paths, val_loader, device)

    print(f"Best Epoch: {best_epoch}")
    print("All Metrics:", all_metrics)
