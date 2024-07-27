import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import sys
from tqdm import tqdm
import lpips
from DISTS_pytorch import DISTS
import time
from plot import save_dists, save_inference_time_plot, generate_comparison_plot, save_example_images, save_heatmaps, save_histograms_of_differences, save_frequency_domain_analysis, save_frequency_domain_analysis_multiple_epochs, plot_psd_comparison, save_metrics

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

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
                        print(f"Original degraded_np shape: {degraded_np.shape}")
                        
                        if degraded_np.ndim == 3 and degraded_np.shape[0] == 3:
                            degraded_np = np.transpose(degraded_np, (1, 2, 0))
                            print(f"Transposed degraded_np shape: {degraded_np.shape}")
                        
                        if degraded_np.ndim == 3 and degraded_np.shape[2] == 3:  # Convert RGB to grayscale
                            degraded_np_2 = np.mean(degraded_np, axis=2)
                            print(f"Grayscale degraded_np shape: {degraded_np.shape}")
                        
                        if gt_image_np.ndim == 3 and gt_image_np.shape[0] == 3:
                            gt_image_np = np.transpose(gt_image_np, (1, 2, 0))
                            print(f"Transposed gt_image_np shape: {gt_image_np.shape}")
                        
                        if gt_image_np.ndim == 3 and gt_image_np.shape[2] == 3:  # Convert RGB to grayscale
                            gt_image_np_2 = np.mean(gt_image_np, axis=2)
                            print(f"Grayscale gt_image_np shape: {gt_image_np.shape}")

                        if degraded_np.shape != gt_image_np.shape:
                            print(f"degraded_np shape: {degraded_np.shape}")
                            print(f"gt_image_np shape: {gt_image_np.shape}")
                            raise ValueError("Input images must have the same dimensions")

                        if degraded_np.shape[0] < 8 or degraded_np.shape[1] < 8:
                            raise ValueError("Image is too small for BM3D processing")

                        print("Applying BM3D denoising...")
                        bm3d_denoised = bm3d.bm3d(degraded_np_2, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)
                        print("BM3D denoising applied successfully.")
                        
                        # Convert bm3d_denoised to float32
                        bm3d_denoised = bm3d_denoised.astype(np.float32)

                        psnr_bm3d = calculate_psnr(gt_image_np_2, bm3d_denoised, data_range=1.0)
                        ssim_bm3d = calculate_ssim(gt_image_np_2, bm3d_denoised)
                        bm3d_denoised_tensor = torch.tensor(bm3d_denoised).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
                        
                        gt_image_tensor = torch.tensor(gt_image_np).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
                        lpips_bm3d = lpips_model(normalize_to_neg1_1(gt_image_tensor.repeat(1, 3, 1, 1)), normalize_to_neg1_1(bm3d_denoised_tensor.repeat(1, 3, 1, 1))).item()
                        dists_bm3d = dists_model(normalize_to_neg1_1(gt_image_tensor.repeat(1, 3, 1, 1)), normalize_to_neg1_1(bm3d_denoised_tensor.repeat(1, 3, 1, 1))).item()
                    except ValueError as e:
                        print(f"BM3D Error: {e}")
                        psnr_bm3d = 0
                        ssim_bm3d = 0
                        lpips_bm3d = 0
                        dists_bm3d = 0
                    except Exception as e:
                        print(f"Unexpected error during BM3D processing: {e}")
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    image_folder = 'dataset/DIV2K_valid_HR.nosync'
    train_noise_levels = [10, 20, 30, 40, 50]
    val_noise_levels = [10, 20, 30, 40, 50]

    train_loader, val_loader = load_data(image_folder, batch_size=1, num_workers=8, validation_split=0.5, augment=False, dataset_percentage=0.01, only_validation=False, include_noise_level=True, train_noise_levels=train_noise_levels, val_noise_levels=val_noise_levels, use_rgb=True)

    epochs_to_evaluate = [202]
    diffusion_model_paths = [f"checkpoints/diffusion_RDUnet_model_checkpointed_epoch_{epoch}.pth" for epoch in epochs_to_evaluate]
    unet_model_path = "checkpoints/rdunet_denoising.pth"

    selected_studies = ['metrics', 'dists', 'example_images', 'histograms_of_differences', 'heatmaps', 'frequency_domain_analysis']
    save_directory = 'evaluation_results'

    evaluate_model_and_plot(epochs_to_evaluate, diffusion_model_paths, unet_model_path, val_loader=val_loader, device=device, include_noise_level=True, use_bm3d=False, save_dir=save_directory, studies=selected_studies)
