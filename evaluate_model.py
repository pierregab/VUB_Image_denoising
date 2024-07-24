import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import sys
from tqdm import tqdm
import lpips
from DISTS_pytorch import DISTS
from scipy.signal import welch
import time

from dataset_creation.data_loader import load_data
from UNet.RDUNet_model import RDUNet
from diffusion_denoising.diffusion_RDUnet import DiffusionModel, RDUNet_T

# Set high dpi for matplotlib
plt.rcParams['figure.dpi'] = 300

def denormalize(tensor, mean=0.5, std=0.5):
    return tensor * std + mean

def normalize_to_neg1_1(tensor):
    return 2 * tensor - 1

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

def calculate_mae(X, Y):
    return np.mean(np.abs(X - Y))

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

def save_example_images(example_images, save_dir):
    noise_levels_to_plot = [15, 30, 50]
    filtered_images = {k: v for k, v in example_images.items() if k in noise_levels_to_plot}

    num_levels = len(filtered_images)
    if num_levels == 0:
        print("No example images to plot.")
        return

    fig, axs = plt.subplots(num_levels, 4, figsize=(20, 5 * num_levels))

    for i, (sigma, images) in enumerate(filtered_images.items()):
        gt_image, degraded_image, predicted_unet_image, predicted_diffusion_image = images

        # Ensure all images are in the shape (H, W, C) and in the range [0, 1]
        gt_image = np.transpose(gt_image, (1, 2, 0))
        degraded_image = np.transpose(degraded_image, (1, 2, 0))
        predicted_unet_image = np.transpose(predicted_unet_image, (1, 2, 0))
        predicted_diffusion_image = np.transpose(predicted_diffusion_image, (1, 2, 0))

        axs[i, 0].imshow(gt_image)
        axs[i, 0].set_title(f'Ground Truth (Sigma: {sigma})')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(degraded_image)
        axs[i, 1].set_title('Noisy')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(predicted_unet_image)
        axs[i, 2].set_title('Denoised (UNet)')
        axs[i, 2].axis('off')

        axs[i, 3].imshow(predicted_diffusion_image)
        axs[i, 3].set_title('Denoised (Diffusion)')
        axs[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'example_images.png'))
    plt.close()

def save_error_map(gt_image, predicted_image, save_dir):
    error_map = np.abs(gt_image - predicted_image)
    plt.imshow(error_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Error Map')
    plt.savefig(os.path.join(save_dir, 'error_map.png'))
    plt.close()

def save_histograms_of_differences(example_images, last_epoch, save_dir):
    noise_levels_to_plot = [15, 30, 50]
    filtered_images = {k: v for k, v in example_images.items() if k[1] in noise_levels_to_plot and k[0] == last_epoch}

    num_levels = len(filtered_images)
    if num_levels == 0:
        print("No example images to plot.")
        return

    fig, axs = plt.subplots(num_levels, 2, figsize=(20, 5 * num_levels))

    for i, ((epoch, sigma), images) in enumerate(filtered_images.items()):
        gt_image, degraded_image, predicted_unet_image, predicted_diffusion_image = images

        differences_unet = (gt_image - predicted_unet_image).flatten()
        differences_diffusion = (gt_image - predicted_diffusion_image).flatten()

        axs[i, 0].hist(differences_unet, bins=50, color='blue', alpha=0.7)
        axs[i, 0].set_title(f'Histogram of Differences (UNet) - Epoch: {epoch}, Sigma: {sigma}')
        axs[i, 0].set_xlabel('Difference')
        axs[i, 0].set_ylabel('Frequency')

        axs[i, 1].hist(differences_diffusion, bins=50, color='green', alpha=0.7)
        axs[i, 1].set_title(f'Histogram of Differences (Diffusion) - Epoch: {epoch}, Sigma: {sigma}')
        axs[i, 1].set_xlabel('Difference')
        axs[i, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'histograms_of_differences.png'))
    plt.close()

def save_heatmaps(aggregated_diff_map_unet, aggregated_diff_map_diffusion, save_dir):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    if aggregated_diff_map_unet.ndim == 3:
        aggregated_diff_map_unet = np.mean(aggregated_diff_map_unet, axis=0)
    if aggregated_diff_map_diffusion.ndim == 3:
        aggregated_diff_map_diffusion = np.mean(aggregated_diff_map_diffusion, axis=0)

    vmin = min(aggregated_diff_map_unet.min(), aggregated_diff_map_diffusion.min())
    vmax = max(aggregated_diff_map_unet.max(), aggregated_diff_map_diffusion.max())

    im_unet = axs[0].imshow(aggregated_diff_map_unet, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    axs[0].set_title('Aggregated Difference Map (UNet)')
    fig.colorbar(im_unet, ax=axs[0], orientation='vertical')

    im_diffusion = axs[1].imshow(aggregated_diff_map_diffusion, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    axs[1].set_title('Aggregated Difference Map (Diffusion)')
    fig.colorbar(im_diffusion, ax=axs[1], orientation='vertical')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'heatmaps.png'))
    plt.close()

def save_frequency_domain_analysis(metrics, last_epoch, save_dir, high_freq_threshold=0.5):
    epochs = sorted(set(metrics['epoch']))
    noise_levels = np.array(metrics['noise_level'])
    unique_noise_levels = sorted(np.unique(noise_levels))
    avg_mae_diff_unet = []
    avg_mae_diff_diffusion = []

    for nl in unique_noise_levels:
        idx = (noise_levels == nl) & (np.array(metrics['epoch']) == last_epoch)
        mae_diff_unet = []
        mae_diff_diffusion = []

        for gt_image, predicted_unet_image, predicted_diffusion_image in zip(
            np.array(metrics['gt_image'])[idx],
            np.array(metrics['predicted_unet_image'])[idx],
            np.array(metrics['predicted_diffusion_image'])[idx]
        ):
            gt_image = gt_image.squeeze()
            predicted_unet_image = predicted_unet_image.squeeze()
            predicted_diffusion_image = predicted_diffusion_image.squeeze()

            f, Pxx_gt = welch(gt_image.flatten(), nperseg=256)
            _, Pxx_unet = welch(predicted_unet_image.flatten(), nperseg=256)
            _, Pxx_diffusion = welch(predicted_diffusion_image.flatten(), nperseg=256)

            high_freq_idx = f >= high_freq_threshold * np.max(f)

            Pxx_gt_high = Pxx_gt[high_freq_idx]
            Pxx_unet_high = Pxx_unet[high_freq_idx]
            Pxx_diffusion_high = Pxx_diffusion[high_freq_idx]

            mae_diff_unet.append(np.mean(np.abs(Pxx_gt_high - Pxx_unet_high)))
            mae_diff_diffusion.append(np.mean(np.abs(Pxx_gt_high - Pxx_diffusion_high)))

        avg_mae_diff_unet.append(np.mean(mae_diff_unet))
        avg_mae_diff_diffusion.append(np.mean(mae_diff_diffusion))

    plt.figure(figsize=(10, 6))
    plt.plot(unique_noise_levels, avg_mae_diff_unet, 'o-', label='UNet Model', color='purple')
    plt.plot(unique_noise_levels, avg_mae_diff_diffusion, 'o-', label=f'Diffusion Model (Epoch {last_epoch})', color='green')
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('MAE in High-Frequency Domain')
    plt.title('MAE in High-Frequency Domain Analysis')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'high_frequency_domain_analysis.png'))
    plt.close()

def save_frequency_domain_analysis_multiple_epochs(metrics, epochs, save_dir, high_freq_threshold=0.5):
    noise_levels = np.array(metrics['noise_level'])
    unique_noise_levels = sorted(np.unique(noise_levels))

    avg_mae_diff_unet = []
    avg_mae_diff_diffusion = {epoch: [] for epoch in epochs}

    for nl in unique_noise_levels:
        idx = (noise_levels == nl)
        mae_diff_unet = []
        mae_diff_diffusion = {epoch: [] for epoch in epochs}

        for gt_image, predicted_unet_image, epoch, predicted_diffusion_image in zip(
            np.array(metrics['gt_image'])[idx],
            np.array(metrics['predicted_unet_image'])[idx],
            np.array(metrics['epoch'])[idx],
            np.array(metrics['predicted_diffusion_image'])[idx]
        ):
            gt_image = gt_image.squeeze()
            predicted_unet_image = predicted_unet_image.squeeze()
            predicted_diffusion_image = predicted_diffusion_image.squeeze()

            f, Pxx_gt = welch(gt_image.flatten(), nperseg=256)
            _, Pxx_unet = welch(predicted_unet_image.flatten(), nperseg=256)
            _, Pxx_diffusion = welch(predicted_diffusion_image.flatten(), nperseg=256)

            high_freq_idx = f >= high_freq_threshold * np.max(f)

            Pxx_gt_high = Pxx_gt[high_freq_idx]
            Pxx_unet_high = Pxx_unet[high_freq_idx]
            Pxx_diffusion_high = Pxx_diffusion[high_freq_idx]

            mae_diff_unet.append(np.mean(np.abs(Pxx_gt_high - Pxx_unet_high)))
            mae_diff_diffusion[epoch].append(np.mean(np.abs(Pxx_gt_high - Pxx_diffusion_high)))

        avg_mae_diff_unet.append(np.mean(mae_diff_unet))
        for epoch in epochs:
            avg_mae_diff_diffusion[epoch].append(np.mean(mae_diff_diffusion[epoch]))

    plt.figure(figsize=(10, 6))
    plt.plot(unique_noise_levels, avg_mae_diff_unet, 'o-', label='UNet Model', color='purple')
    colors = ['green', 'blue', 'orange', 'red', 'black', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    for idx, epoch in enumerate(epochs):
        plt.plot(unique_noise_levels, avg_mae_diff_diffusion[epoch], 'o-', label=f'Diffusion Model (Epoch {epoch})', color=colors[idx % len(colors)])
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('MAE in High-Frequency Domain')
    plt.title('MAE in High-Frequency Domain Analysis')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'high_frequency_domain_analysis_multiple_epochs.png'))
    plt.close()

def plot_psd_comparison(metrics, last_epoch, save_dir, high_freq_threshold=0.5):
    epochs = sorted(set(metrics['epoch']))
    noise_levels = np.array(metrics['noise_level'])
    unique_noise_levels = sorted(np.unique(noise_levels))

    for nl in unique_noise_levels:
        idx = (noise_levels == nl) & (np.array(metrics['epoch']) == last_epoch)
        
        psd_gt_all = []
        psd_unet_all = []
        psd_diffusion_all = []
        psd_degraded_all = []
        
        for gt_image, degraded_image, predicted_unet_image, predicted_diffusion_image in zip(
            np.array(metrics['gt_image'])[idx],
            np.array(metrics['degraded_image'])[idx],
            np.array(metrics['predicted_unet_image'])[idx],
            np.array(metrics['predicted_diffusion_image'])[idx]
        ):
            gt_image = gt_image.squeeze()
            degraded_image = degraded_image.squeeze()
            predicted_unet_image = predicted_unet_image.squeeze()
            predicted_diffusion_image = predicted_diffusion_image.squeeze()

            f_gt, Pxx_gt = welch(gt_image.flatten(), nperseg=256)
            _, Pxx_degraded = welch(degraded_image.flatten(), nperseg=256)
            _, Pxx_unet = welch(predicted_unet_image.flatten(), nperseg=256)
            _, Pxx_diffusion = welch(predicted_diffusion_image.flatten(), nperseg=256)

            high_freq_idx = f_gt >= high_freq_threshold * np.max(f_gt)

            psd_gt_all.append(Pxx_gt[high_freq_idx])
            psd_degraded_all.append(Pxx_degraded[high_freq_idx])
            psd_unet_all.append(Pxx_unet[high_freq_idx])
            psd_diffusion_all.append(Pxx_diffusion[high_freq_idx])
        
        avg_psd_gt = np.mean(psd_gt_all, axis=0)
        avg_psd_degraded = np.mean(psd_degraded_all, axis=0)
        avg_psd_unet = np.mean(psd_unet_all, axis=0)
        avg_psd_diffusion = np.mean(psd_diffusion_all, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(f_gt[high_freq_idx], avg_psd_gt, label='Ground Truth', color='black')
        plt.plot(f_gt[high_freq_idx], avg_psd_degraded, label='Degraded', color='red')
        plt.plot(f_gt[high_freq_idx], avg_psd_unet, label='UNet Model', color='purple')
        plt.plot(f_gt[high_freq_idx], avg_psd_diffusion, label=f'Diffusion Model (Epoch {last_epoch})', color='green')
        plt.xlabel('Frequency')
        plt.ylabel('Power Spectral Density (Log Scale)')
        plt.yscale('log')
        plt.title(f'Average PSD Comparison at Noise Level {nl}')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'psd_comparison_high_freq_noise_level_{nl}.png'))
        plt.close()

def evaluate_model_and_plot(epochs, diffusion_model_paths, unet_model_path, val_loader, device, include_noise_level=False, use_bm3d=False, save_dir='results', studies=None):
    if use_bm3d:
        import bm3d

    os.makedirs(save_dir, exist_ok=True)

    lpips_model = lpips.LPIPS(net='alex').to(device)
    dists_model = DISTS().to(device)

    metrics = {'epoch': [], 'noise_level': [], 'psnr_degraded': [], 'psnr_diffusion': [], 'psnr_unet': [], 'psnr_bm3d': [], 'ssim_degraded': [], 'ssim_diffusion': [], 'ssim_unet': [], 'ssim_bm3d': [], 'lpips_degraded': [], 'lpips_diffusion': [], 'lpips_unet': [], 'lpips_bm3d': [], 'dists_degraded': [], 'dists_diffusion': [], 'dists_unet': [], 'dists_bm3d': [], 'gt_image': [], 'degraded_image': [], 'predicted_unet_image': [], 'predicted_diffusion_image': []}
    example_images = {}
    aggregated_diff_map_unet = None
    aggregated_diff_map_diffusion = None
    count = 0

    inference_times = {'unet': [], 'diffusion': []}

    evaluate_unet = os.path.exists(unet_model_path)
    if evaluate_unet:
        unet_model = RDUNet(base_filters=128).to(device)  # Updated to use RDUNet
        unet_checkpoint = torch.load(unet_model_path, map_location=device)
        if 'model_state_dict' in unet_checkpoint:
            unet_model.load_state_dict(unet_checkpoint['model_state_dict'])
        else:
            unet_model.load_state_dict(unet_checkpoint)
        unet_model.eval()
    else:
        print(f"UNet model path '{unet_model_path}' does not exist. Skipping UNet evaluation.")

    for epoch, diffusion_model_path in zip(epochs, diffusion_model_paths):
        diffusion_model = DiffusionModel(RDUNet_T(base_filters=32).to(device)).to(device)
        diffusion_checkpoint = torch.load(diffusion_model_path, map_location=device)
        if isinstance(diffusion_checkpoint, dict) and 'model_state_dict' in diffusion_checkpoint:
            diffusion_model.load_state_dict(diffusion_checkpoint['model_state_dict'])
        else:
            diffusion_model = diffusion_checkpoint
        diffusion_model.to(device)
        diffusion_model.eval()

        for data in tqdm(val_loader, desc=f"Evaluating Epoch {epoch}"):
            if include_noise_level:
                degraded_images, gt_images, noise_levels = data
            else:
                degraded_images, gt_images = data
                noise_levels = None

            degraded_images = degraded_images.to(device)
            gt_images = gt_images.to(device)

            with torch.no_grad():
                # Measure inference time for the diffusion model
                start_time = time.time()
                t = torch.randint(0, diffusion_model.timesteps, (1,), device=device).float() / diffusion_model.timesteps
                predicted_diffusions = diffusion_model.improved_sampling(degraded_images)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                diffusion_inference_time = end_time - start_time
                inference_times['diffusion'].append(diffusion_inference_time)

                if evaluate_unet:
                    # Measure inference time for the UNet model
                    start_time = time.time()
                    predicted_unets = unet_model(degraded_images)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()
                    unet_inference_time = end_time - start_time
                    inference_times['unet'].append(unet_inference_time)

            for j in range(degraded_images.size(0)):
                gt_image = gt_images[j]
                degraded_image = degraded_images[j]
                predicted_diffusion = predicted_diffusions[j]
                if evaluate_unet:
                    predicted_unet = predicted_unets[j]

                psnr_degraded, ssim_degraded, lpips_degraded, dists_degraded = compute_metrics(gt_image, degraded_image, lpips_model, dists_model, use_rgb=True)
                psnr_diffusion, ssim_diffusion, lpips_diffusion, dists_diffusion = compute_metrics(gt_image, predicted_diffusion, lpips_model, dists_model, use_rgb=True)

                if evaluate_unet:
                    psnr_unet, ssim_unet, lpips_unet, dists_unet = compute_metrics(gt_image, predicted_unet, lpips_model, dists_model)

                degraded_np = denormalize(degraded_image.cpu().numpy().squeeze())
                gt_image_np = denormalize(gt_image.cpu().numpy().squeeze())
                predicted_diffusion_np = denormalize(predicted_diffusion.cpu().numpy().squeeze())
                if evaluate_unet:
                    predicted_unet_np = denormalize(predicted_unet.cpu().numpy().squeeze())

                if use_bm3d:
                    try:
                        # Ensure the degraded_np image is in the correct format for BM3D
                        if degraded_np.ndim == 3 and degraded_np.shape[0] == 3:
                            degraded_np = np.transpose(degraded_np, (1, 2, 0))
                        if degraded_np.ndim == 3 and degraded_np.shape[2] == 3:  # Convert RGB to grayscale
                            degraded_np = np.mean(degraded_np, axis=2)
                        
                        if degraded_np.shape[0] < 8 or degraded_np.shape[1] < 8:
                            raise ValueError("Image is too small for BM3D processing")

                        bm3d_denoised = bm3d.bm3d(degraded_np, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)
                        psnr_bm3d = calculate_psnr(gt_image_np, bm3d_denoised, data_range=1.0)
                        ssim_bm3d = calculate_ssim(gt_image_np, bm3d_denoised)
                        bm3d_denoised_tensor = torch.tensor(bm3d_denoised).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
                        lpips_bm3d = lpips_model(normalize_to_neg1_1(gt_image.unsqueeze(0).repeat(3, 1, 1, 1)), normalize_to_neg1_1(bm3d_denoised_tensor)).item()
                        dists_bm3d = dists_model(normalize_to_neg1_1(gt_image.unsqueeze(0).repeat(3, 1, 1, 1)), normalize_to_neg1_1(bm3d_denoised_tensor)).item()
                    except ValueError as e:
                        print(f"BM3D Error: {e}")
                        psnr_bm3d = 0
                        ssim_bm3d = 0
                        lpips_bm3d = 0
                        dists_bm3d = 0
                else:
                    psnr_bm3d = 0
                    ssim_bm3d = 0
                    lpips_bm3d = 0
                    dists_bm3d = 0

                metrics['epoch'].append(epoch)
                metrics['noise_level'].append(noise_levels[j].item() if noise_levels is not None else 0)
                metrics['psnr_degraded'].append(psnr_degraded)
                metrics['psnr_diffusion'].append(psnr_diffusion)
                if evaluate_unet:
                    metrics['psnr_unet'].append(psnr_unet)
                else:
                    metrics['psnr_unet'].append(0)
                metrics['psnr_bm3d'].append(psnr_bm3d)
                metrics['ssim_degraded'].append(ssim_degraded)
                metrics['ssim_diffusion'].append(ssim_diffusion)
                if evaluate_unet:
                    metrics['ssim_unet'].append(ssim_unet)
                else:
                    metrics['ssim_unet'].append(0)
                metrics['ssim_bm3d'].append(ssim_bm3d)
                metrics['lpips_degraded'].append(lpips_degraded)
                metrics['lpips_diffusion'].append(lpips_diffusion)
                if evaluate_unet:
                    metrics['lpips_unet'].append(lpips_unet)
                else:
                    metrics['lpips_unet'].append(0)
                metrics['lpips_bm3d'].append(lpips_bm3d)
                metrics['dists_degraded'].append(dists_degraded)
                metrics['dists_diffusion'].append(dists_diffusion)
                if evaluate_unet:
                    metrics['dists_unet'].append(dists_unet)
                else:
                    metrics['dists_unet'].append(0)
                metrics['dists_bm3d'].append(dists_bm3d)
                metrics['gt_image'].append(gt_image_np)
                metrics['degraded_image'].append(degraded_np)
                if evaluate_unet:
                    metrics['predicted_unet_image'].append(predicted_unet_np)
                else:
                    metrics['predicted_unet_image'].append(np.zeros_like(gt_image_np))
                metrics['predicted_diffusion_image'].append(predicted_diffusion_np)

                if aggregated_diff_map_unet is None:
                    aggregated_diff_map_unet = np.abs(gt_image_np - (predicted_unet_np if evaluate_unet else np.zeros_like(gt_image_np)))
                    aggregated_diff_map_diffusion = np.abs(gt_image_np - predicted_diffusion_np)
                else:
                    if evaluate_unet:
                        aggregated_diff_map_unet += np.abs(gt_image_np - predicted_unet_np)
                    aggregated_diff_map_diffusion += np.abs(gt_image_np - predicted_diffusion_np)
                count += 1

                sigma_level = int(noise_levels[j].item()) if noise_levels is not None else 0
                if sigma_level in [15, 30, 50]:
                    if evaluate_unet:
                        example_images[(epoch, sigma_level)] = (gt_image_np, degraded_np, predicted_unet_np, predicted_diffusion_np)
                    else:
                        example_images[(epoch, sigma_level)] = (gt_image_np, degraded_np, np.zeros_like(gt_image_np), predicted_diffusion_np)

    aggregated_diff_map_unet /= count
    aggregated_diff_map_diffusion /= count

    if studies is None or 'metrics' in studies:
        save_metrics(metrics, epochs[-1], use_bm3d, save_dir)
    if studies is None or 'dists' in studies:
        save_dists(metrics, epochs[-1], save_dir)
    if example_images:
        if studies is None or 'example_images' in studies:
            save_example_images({key[1]: value for key, value in example_images.items()}, save_dir)
        if studies is None or 'histograms_of_differences' in studies:
            save_histograms_of_differences(example_images, epochs[-1], save_dir)
    else:
        print("No example images to plot.")
    if studies is None or 'heatmaps' in studies:
        save_heatmaps(aggregated_diff_map_unet, aggregated_diff_map_diffusion, save_dir)
    if studies is None or 'frequency_domain_analysis' in studies:
        save_frequency_domain_analysis(metrics, epochs[-1], save_dir)
    if len(epochs) > 1:
        save_frequency_domain_analysis_multiple_epochs(metrics, epochs, save_dir)
    
    # Call the PSD comparison plotting
    plot_psd_comparison(metrics, epochs[-1], save_dir)

    # Save the inference time comparison plot
    save_inference_time_plot(inference_times, save_dir)

    # Generate the comparison plot
    generate_comparison_plot(metrics, epochs, save_dir)


def save_metrics(metrics, last_epoch, use_bm3d, save_dir):
    epochs = sorted(set(metrics['epoch']))
    noise_levels = np.array(metrics['noise_level'])
    psnr_degraded = np.array(metrics['psnr_degraded'])
    psnr_diffusion = np.array(metrics['psnr_diffusion'])
    psnr_unet = np.array(metrics['psnr_unet'])
    psnr_bm3d = np.array(metrics['psnr_bm3d'])
    ssim_degraded = np.array(metrics['ssim_degraded'])
    ssim_diffusion = np.array(metrics['ssim_diffusion'])
    ssim_unet = np.array(metrics['ssim_unet'])
    ssim_bm3d = np.array(metrics['ssim_bm3d'])
    lpips_degraded = np.array(metrics['lpips_degraded'])
    lpips_diffusion = np.array(metrics['lpips_diffusion'])
    lpips_unet = np.array(metrics['lpips_unet'])
    lpips_bm3d = np.array(metrics['lpips_bm3d'])

    unique_noise_levels = sorted(np.unique(noise_levels))

    avg_psnr_degraded = [np.mean(psnr_degraded[noise_levels == nl]) for nl in unique_noise_levels]
    avg_psnr_diffusion_last = [np.mean(psnr_diffusion[(noise_levels == nl) & (np.array(metrics['epoch']) == last_epoch)]) for nl in unique_noise_levels]
    avg_psnr_unet = [np.mean(psnr_unet[noise_levels == nl]) for nl in unique_noise_levels]
    avg_psnr_bm3d = [np.mean(psnr_bm3d[noise_levels == nl]) for nl in unique_noise_levels] if use_bm3d else []
    avg_ssim_degraded = [np.mean(ssim_degraded[noise_levels == nl]) for nl in unique_noise_levels]
    avg_ssim_diffusion_last = [np.mean(ssim_diffusion[(noise_levels == nl) & (np.array(metrics['epoch']) == last_epoch)]) for nl in unique_noise_levels]
    avg_ssim_unet = [np.mean(ssim_unet[noise_levels == nl]) for nl in unique_noise_levels]
    avg_ssim_bm3d = [np.mean(ssim_bm3d[noise_levels == nl]) for nl in unique_noise_levels] if use_bm3d else []
    avg_lpips_degraded = [np.mean(lpips_degraded[noise_levels == nl]) for nl in unique_noise_levels]
    avg_lpips_diffusion_last = [np.mean(lpips_diffusion[(noise_levels == nl) & (np.array(metrics['epoch']) == last_epoch)]) for nl in unique_noise_levels]
    avg_lpips_unet = [np.mean(lpips_unet[noise_levels == nl]) for nl in unique_noise_levels]
    avg_lpips_bm3d = [np.mean(lpips_bm3d[noise_levels == nl]) for nl in unique_noise_levels] if use_bm3d else []

    fig, axs = plt.subplots(3, 2, figsize=(20, 18))

    axs[0, 0].plot(unique_noise_levels, avg_psnr_degraded, 'o-', label='Degraded', color='red')
    axs[0, 0].plot(unique_noise_levels, avg_psnr_unet, 'o-', label='UNet Model', color='purple')
    axs[0, 0].plot(unique_noise_levels, avg_psnr_diffusion_last, 'o-', label=f'Diffusion Model (Epoch {last_epoch})', color='green')
    if use_bm3d:
        axs[0, 0].plot(unique_noise_levels, avg_psnr_bm3d, 'o-', label='BM3D', color='blue')
    axs[0, 0].set_xlabel('Noise Standard Deviation')
    axs[0, 0].set_ylabel('PSNR')
    axs[0, 0].set_title('PSNR value variation curve')
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[1, 0].plot(unique_noise_levels, avg_ssim_degraded, 'o-', label='Degraded', color='red')
    axs[1, 0].plot(unique_noise_levels, avg_ssim_unet, 'o-', label='UNet Model', color='purple')
    axs[1, 0].plot(unique_noise_levels, avg_ssim_diffusion_last, 'o-', label=f'Diffusion Model (Epoch {last_epoch})', color='green')
    if use_bm3d:
        axs[1, 0].plot(unique_noise_levels, avg_ssim_bm3d, 'o-', label='BM3D', color='blue')
    axs[1, 0].set_xlabel('Noise Standard Deviation')
    axs[1, 0].set_ylabel('SSIM')
    axs[1, 0].set_title('SSIM value variation curve')
    axs[1, 0].legend()
    axs[1, 0].grid()

    axs[2, 0].plot(unique_noise_levels, avg_lpips_degraded, 'o-', label='Degraded', color='red')
    axs[2, 0].plot(unique_noise_levels, avg_lpips_unet, 'o-', label='UNet Model', color='purple')
    axs[2, 0].plot(unique_noise_levels, avg_lpips_diffusion_last, 'o-', label=f'Diffusion Model (Epoch {last_epoch})', color='green')
    if use_bm3d:
        axs[2, 0].plot(unique_noise_levels, avg_lpips_bm3d, 'o-', label='BM3D', color='blue')
    axs[2, 0].set_xlabel('Noise Standard Deviation')
    axs[2, 0].set_ylabel('LPIPS')
    axs[2, 0].set_title('LPIPS value variation curve')
    axs[2, 0].legend()
    axs[2, 0].grid()

    colors = ['blue', 'orange', 'cyan', 'magenta', 'black', 'yellow', 'green', 'red']
    for idx, epoch in enumerate(epochs):
        epoch_indices = [i for i, e in enumerate(metrics['epoch']) if e == epoch]
        unique_noise_levels = sorted(np.unique(noise_levels[epoch_indices]))

        avg_psnr_diffusion = [np.mean(psnr_diffusion[epoch_indices][noise_levels[epoch_indices] == nl]) for nl in unique_noise_levels]
        avg_ssim_diffusion = [np.mean(ssim_diffusion[epoch_indices][noise_levels[epoch_indices] == nl]) for nl in unique_noise_levels]
        avg_lpips_diffusion = [np.mean(lpips_diffusion[epoch_indices][noise_levels[epoch_indices] == nl]) for nl in unique_noise_levels]

        axs[0, 1].plot(unique_noise_levels, avg_psnr_diffusion, 'o-', label=f'Diffusion Model (Epoch {epoch})', color=colors[idx % len(colors)])
        axs[1, 1].plot(unique_noise_levels, avg_ssim_diffusion, 'o-', label=f'Diffusion Model (Epoch {epoch})', color=colors[idx % len(colors)])
        axs[2, 1].plot(unique_noise_levels, avg_lpips_diffusion, 'o-', label=f'Diffusion Model (Epoch {epoch})', color=colors[idx % len(colors)])

    axs[0, 1].set_xlabel('Noise Standard Deviation')
    axs[0, 1].set_ylabel('PSNR')
    axs[0, 1].set_title('PSNR value variation curve (Diffusion Model)')
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[1, 1].set_xlabel('Noise Standard Deviation')
    axs[1, 1].set_ylabel('SSIM')
    axs[1, 1].set_title('SSIM value variation curve (Diffusion Model)')
    axs[1, 1].legend()
    axs[1, 1].grid()

    axs[2, 1].set_xlabel('Noise Standard Deviation')
    axs[2, 1].set_ylabel('LPIPS')
    axs[2, 1].set_title('LPIPS value variation curve (Diffusion Model)')
    axs[2, 1].legend()
    axs[2, 1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics.png'))
    plt.close()

def save_dists(metrics, last_epoch, save_dir):
    epochs = sorted(set(metrics['epoch']))
    noise_levels = np.array(metrics['noise_level'])
    dists_degraded = np.array(metrics['dists_degraded'])
    dists_diffusion = np.array(metrics['dists_diffusion'])
    dists_unet = np.array(metrics['dists_unet'])
    dists_bm3d = np.array(metrics['dists_bm3d'])

    unique_noise_levels = sorted(np.unique(noise_levels))

    avg_dists_degraded = [np.mean(dists_degraded[noise_levels == nl]) for nl in unique_noise_levels]
    avg_dists_diffusion_last = [np.mean(dists_diffusion[(noise_levels == nl) & (np.array(metrics['epoch']) == last_epoch)]) for nl in unique_noise_levels]
    avg_dists_unet = [np.mean(dists_unet[noise_levels == nl]) for nl in unique_noise_levels]
    avg_dists_bm3d = [np.mean(dists_bm3d[noise_levels == nl]) for nl in unique_noise_levels]

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))

    axs.plot(unique_noise_levels, avg_dists_degraded, 'o-', label='Degraded', color='red')
    axs.plot(unique_noise_levels, avg_dists_unet, 'o-', label='UNet Model', color='purple')
    axs.plot(unique_noise_levels, avg_dists_diffusion_last, 'o-', label=f'Diffusion Model (Epoch {last_epoch})', color='green')
    axs.plot(unique_noise_levels, avg_dists_bm3d, 'o-', label='BM3D', color='blue')

    axs.set_xlabel('Noise Standard Deviation')
    axs.set_ylabel('DISTS')
    axs.set_title('DISTS value variation curve')
    axs.legend()
    axs.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dists.png'))
    plt.close()

def save_inference_time_plot(inference_times, save_dir):
    avg_inference_time_unet = np.mean(inference_times['unet'])
    avg_inference_time_diffusion = np.mean(inference_times['diffusion'])

    labels = ['UNet', 'Diffusion']
    times = [avg_inference_time_unet, avg_inference_time_diffusion]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, times, color=['purple', 'green'])
    plt.ylabel('Average Inference Time (s)')
    plt.title('Average Inference Time Comparison')
    plt.savefig(os.path.join(save_dir, 'inference_time_comparison.png'))
    plt.close()

def generate_comparison_plot(metrics, epochs, save_dir):
    noise_levels = np.array(metrics['noise_level'])
    unique_noise_levels = sorted(np.unique(noise_levels))
    
    psnr_diffusion = np.array(metrics['psnr_diffusion'])
    psnr_unet = np.array(metrics['psnr_unet'])
    psnr_bm3d = np.array(metrics['psnr_bm3d'])
    lpips_diffusion = np.array(metrics['lpips_diffusion'])
    lpips_unet = np.array(metrics['lpips_unet'])
    lpips_bm3d = np.array(metrics['lpips_bm3d'])

    avg_psnr_diffusion = [np.mean(psnr_diffusion[noise_levels == nl]) for nl in unique_noise_levels]
    avg_psnr_unet = [np.mean(psnr_unet[noise_levels == nl]) for nl in unique_noise_levels]
    avg_psnr_bm3d = [np.mean(psnr_bm3d[noise_levels == nl]) for nl in unique_noise_levels]
    avg_lpips_diffusion = [np.mean(lpips_diffusion[noise_levels == nl]) for nl in unique_noise_levels]
    avg_lpips_unet = [np.mean(lpips_unet[noise_levels == nl]) for nl in unique_noise_levels]
    avg_lpips_bm3d = [np.mean(lpips_bm3d[noise_levels == nl]) for nl in unique_noise_levels]

    plt.figure(figsize=(10, 6))
    
    # Plot diffusion model
    plt.scatter(avg_lpips_diffusion, avg_psnr_diffusion, label='Diffusion Model', color='green', marker='*', s=100)
    
    # Plot UNet model
    plt.scatter(avg_lpips_unet, avg_psnr_unet, label='UNet Model', color='purple', marker='o', s=100)
    
    # Plot BM3D
    plt.scatter(avg_lpips_bm3d, avg_psnr_bm3d, label='BM3D', color='blue', marker='^', s=100)
    
    plt.xlabel('LPIPS')
    plt.ylabel('PSNR')
    plt.title('Model Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'comparison_plot.png'))
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    image_folder = 'dataset/DIV2K_valid_HR.nosync'
    train_noise_levels = [10, 20, 30, 40, 50, 60, 70, 80]
    val_noise_levels = [10, 20, 30, 40, 50, 60, 70, 80]

    train_loader, val_loader = load_data(image_folder, batch_size=1, num_workers=8, validation_split=0.5, augment=False, dataset_percentage=0.01, only_validation=False, include_noise_level=True, train_noise_levels=train_noise_levels, val_noise_levels=val_noise_levels, use_rgb=True)

    epochs_to_evaluate = [5]
    diffusion_model_paths = [f"checkpoints/diffusion_RDUNet_model_checkpointed_epoch_{epoch}.pth" for epoch in epochs_to_evaluate]
    unet_model_path = "checkpoints/rdunet_denoising.pth"

    selected_studies = ['metrics', 'dists', 'example_images', 'histograms_of_differences', 'heatmaps', 'frequency_domain_analysis']
    save_directory = 'evaluation_results'

    evaluate_model_and_plot(epochs_to_evaluate, diffusion_model_paths, unet_model_path, val_loader=val_loader, device=device, include_noise_level=True, use_bm3d=True, save_dir=save_directory, studies=selected_studies)
