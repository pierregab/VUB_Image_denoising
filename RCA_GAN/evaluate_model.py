import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import sys
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data
from UNet.UNet_model import UNet  # Assuming UNet is defined in this module
from diffusion_denoising.diffusion_model import UNet_S_Checkpointed, DiffusionModel  # Assuming UNet_S_Checkpointed is defined in this module

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

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

def calculate_mae(X, Y):
    return np.mean(np.abs(X - Y))

# Initialize the LPIPS metric
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)

def compute_metrics(original, processed, use_rgb=False):
    original_np = denormalize(original.cpu().numpy().squeeze())
    processed_np = denormalize(processed.cpu().numpy().squeeze())
    
    psnr_value = calculate_psnr(original_np, processed_np, data_range=1.0)
    ssim_value = calculate_ssim(original_np, processed_np, data_range=1.0, use_rgb=use_rgb)
    
    # LPIPS calculation
    original_tensor = original.unsqueeze(0).to(device)
    processed_tensor = processed.unsqueeze(0).to(device)

    # Normalize the input tensors to the range [-1, 1]
    original_tensor = (original_tensor - 0.5) / 0.5
    processed_tensor = (processed_tensor - 0.5) / 0.5

    lpips_value = lpips_metric(original_tensor, processed_tensor).item()
    
    return psnr_value, ssim_value, lpips_value

def plot_example_images(example_images):
    noise_levels_to_plot = [15, 30, 50]
    filtered_images = {k: v for k, v in example_images.items() if k[1] in noise_levels_to_plot}
    
    num_levels = len(filtered_images)
    if num_levels == 0:
        print("No example images to plot.")
        return
    
    fig, axs = plt.subplots(num_levels, 4, figsize=(20, 5 * num_levels))
    
    for i, ((epoch, sigma), images) in enumerate(filtered_images.items()):
        gt_image, degraded_image, predicted_unet_image, predicted_diffusion_image = images
        
        axs[i, 0].imshow(np.transpose(gt_image, (1, 2, 0)))  # Assuming the image format is (C, H, W)
        axs[i, 0].set_title(f'Ground Truth (Sigma: {sigma}, Epoch: {epoch})')
        axs[i, 0].axis('off')
        
        axs[i, 1].imshow(np.transpose(degraded_image, (1, 2, 0)))  # Assuming the image format is (C, H, W)
        axs[i, 1].set_title('Noisy')
        axs[i, 1].axis('off')
        
        axs[i, 2].imshow(predicted_unet_image, cmap='gray')
        axs[i, 2].set_title('Denoised (UNet)')
        axs[i, 2].axis('off')
        
        axs[i, 3].imshow(np.transpose(predicted_diffusion_image, (1, 2, 0)))  # Assuming the image format is (C, H, W)
        axs[i, 3].set_title('Denoised (Diffusion)')
        axs[i, 3].axis('off')
    
    plt.show()

def plot_error_map(gt_image, predicted_image):
    error_map = np.abs(gt_image - predicted_image)
    plt.imshow(error_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Error Map')
    plt.show()

def plot_histograms_of_differences(example_images, last_epoch):
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
    plt.show()

def plot_heatmaps(aggregated_diff_map_unet, aggregated_diff_map_diffusion):
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
    plt.show()

def evaluate_model_and_plot(epochs, diffusion_model_paths, unet_model_path, val_loader, device, include_noise_level=False, use_bm3d=False):
    if use_bm3d:
        import bm3d  # Import bm3d only if use_bm3d is True

    metrics = {'epoch': [], 'noise_level': [], 'psnr_degraded': [], 'psnr_diffusion': [], 'psnr_unet': [], 'psnr_bm3d': [], 'ssim_degraded': [], 'ssim_diffusion': [], 'ssim_unet': [], 'ssim_bm3d': [], 'lpips_degraded': [], 'lpips_diffusion': [], 'lpips_unet': [], 'lpips_bm3d': []}
    example_images = {}
    aggregated_diff_map_unet = None
    aggregated_diff_map_diffusion = None
    count = 0

    # Load the fixed UNet model
    unet_model = UNet().to(device)
    unet_checkpoint = torch.load(unet_model_path, map_location=device)
    if 'model_state_dict' in unet_checkpoint:
        unet_model.load_state_dict(unet_checkpoint['model_state_dict'])
    else:
        unet_model.load_state_dict(unet_checkpoint)
    unet_model.eval()

    for epoch, diffusion_model_path in zip(epochs, diffusion_model_paths):
        # Load the diffusion model checkpoint
        diffusion_model = DiffusionModel(UNet_S_Checkpointed())  # Ensure that DiffusionModel is correctly instantiated
        diffusion_checkpoint = torch.load(diffusion_model_path, map_location=device)
        if isinstance(diffusion_checkpoint, dict) and 'model_state_dict' in diffusion_checkpoint:
            diffusion_model.load_state_dict(diffusion_checkpoint['model_state_dict'])
        else:
            diffusion_model = diffusion_checkpoint
        diffusion_model.to(device)
        diffusion_model.eval()

        for i, data in enumerate(tqdm(val_loader, desc=f"Evaluating Epoch {epoch}")):
            if include_noise_level:
                degraded_image, gt_image, noise_level = data
            else:
                degraded_image, gt_image = data
                noise_level = None

            degraded_image = degraded_image.to(device)
            gt_image = gt_image.to(device)

            with torch.no_grad():
                t = torch.randint(0, diffusion_model.timesteps, (1,), device=device).float() / diffusion_model.timesteps
                predicted_diffusion = diffusion_model.improved_sampling(degraded_image)

                # create single channel image to pass to unet
                degraded_image_1 = degraded_image.mean(dim=1, keepdim=True)
                gt_image_1 = gt_image.mean(dim=1, keepdim=True)

                predicted_unet = unet_model(degraded_image_1)

            for j in range(degraded_image.size(0)):
                psnr_degraded, ssim_degraded, lpips_degraded = compute_metrics(gt_image[j], degraded_image[j], use_rgb=True)
                psnr_diffusion, ssim_diffusion, lpips_diffusion = compute_metrics(gt_image[j], predicted_diffusion[j], use_rgb=True)
                psnr_unet, ssim_unet, lpips_unet = compute_metrics(gt_image_1[j], predicted_unet[j])

                degraded_np = denormalize(degraded_image[j].cpu().numpy().squeeze())
                gt_image_np = denormalize(gt_image[j].cpu().numpy().squeeze())
                predicted_diffusion_np = denormalize(predicted_diffusion[j].cpu().numpy().squeeze())
                predicted_unet_np = denormalize(predicted_unet[j].cpu().numpy().squeeze())
                
                if use_bm3d:
                    bm3d_denoised = bm3d.bm3d(degraded_np, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)
                    psnr_bm3d = calculate_psnr(denormalize(gt_image[j].cpu().numpy().squeeze()), bm3d_denoised, data_range=1.0)
                    ssim_bm3d = calculate_ssim(denormalize(gt_image[j].cpu().numpy().squeeze()), bm3d_denoised)
                    bm3d_tensor = torch.tensor(bm3d_denoised).unsqueeze(0).to(device)
                    lpips_bm3d = lpips_metric(gt_image[j].unsqueeze(0), bm3d_tensor).item()
                else:
                    psnr_bm3d = 0
                    ssim_bm3d = 0
                    lpips_bm3d = 0

                metrics['epoch'].append(epoch)
                metrics['noise_level'].append(noise_level[j].item() if noise_level is not None else 0)
                metrics['psnr_degraded'].append(psnr_degraded)
                metrics['psnr_diffusion'].append(psnr_diffusion)
                metrics['psnr_unet'].append(psnr_unet)
                metrics['psnr_bm3d'].append(psnr_bm3d)
                metrics['ssim_degraded'].append(ssim_degraded)
                metrics['ssim_diffusion'].append(ssim_diffusion)
                metrics['ssim_unet'].append(ssim_unet)
                metrics['ssim_bm3d'].append(ssim_bm3d)
                metrics['lpips_degraded'].append(lpips_degraded)
                metrics['lpips_diffusion'].append(lpips_diffusion)
                metrics['lpips_unet'].append(lpips_unet)
                metrics['lpips_bm3d'].append(lpips_bm3d)

                # Accumulate difference maps
                if aggregated_diff_map_unet is None:
                    aggregated_diff_map_unet = np.abs(gt_image_np - predicted_unet_np)
                    aggregated_diff_map_diffusion = np.abs(gt_image_np - predicted_diffusion_np)
                else:
                    aggregated_diff_map_unet += np.abs(gt_image_np - predicted_unet_np)
                    aggregated_diff_map_diffusion += np.abs(gt_image_np - predicted_diffusion_np)
                count += 1

                sigma_level = int(noise_level[j].item()) if noise_level is not None else 0
                if sigma_level in [15, 30, 50] and (epoch, sigma_level) not in example_images:
                    example_images[(epoch, sigma_level)] = (gt_image_np, degraded_np, predicted_unet_np, predicted_diffusion_np)

    # Average the accumulated difference maps
    aggregated_diff_map_unet /= count
    aggregated_diff_map_diffusion /= count

    plot_metrics(metrics, epochs[-1], use_bm3d)
    print("Example images keys:", example_images.keys())

    # Plot additional histograms and frequency domain analysis
    plot_histograms_of_differences(example_images, epochs[-1])

    # Plot heatmaps of aggregated difference maps
    plot_heatmaps(aggregated_diff_map_unet, aggregated_diff_map_diffusion)

    if example_images:
        plot_example_images({key[1]: value for key, value in example_images.items()})
    else:
        print("No example images to plot.")


def plot_metrics(metrics, last_epoch, use_bm3d):
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

    # Additional plots for different epochs of the diffusion model
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
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    image_folder = 'DIV2K_valid_HR.nosync'
    train_noise_levels = [10, 20, 30, 40, 50, 60, 70, 80]
    val_noise_levels = [10, 20, 30, 40, 50, 60, 70, 80]

    train_loader, val_loader = load_data(image_folder, batch_size=1, num_workers=8, validation_split=0.2, augment=False, dataset_percentage=0.01, only_validation=False, include_noise_level=True, train_noise_levels=train_noise_levels, val_noise_levels=val_noise_levels, use_rgb=True)

    epochs_to_evaluate = [10, 20, 30, 40]
    diffusion_model_paths = [f"checkpoints/diffusion_model_checkpointed_epoch_{epoch}.pth" for epoch in epochs_to_evaluate]
    unet_model_path = "checkpoints/unet_denoising.pth"
    evaluate_model_and_plot(epochs_to_evaluate, diffusion_model_paths, unet_model_path, val_loader=val_loader, device=device, include_noise_level=True, use_bm3d=False)
