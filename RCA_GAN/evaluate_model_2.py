import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, color
from skimage.metrics import structural_similarity as ssim, mean_squared_error, peak_signal_noise_ratio
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from scipy.signal import welch
from DISTS_pytorch import DISTS
import lpips
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import required methods
from dataset_creation.data_loader import load_data
from VUB_Image_denoising.UNet.RDUNet_model import UNet
from diffusion_denoising.diffusion_model import UNet_S_Checkpointed, DiffusionModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Frequency Domain Analysis
def frequency_domain_analysis(images, titles, gt_image):
    plt.figure(figsize=(20, 5))
    for i, img in enumerate(images):
        f, Pxx_den = welch(img.flatten(), nperseg=256)
        if i == 0:
            gt_f, gt_Pxx_den = f, Pxx_den
            plt.semilogy(f, Pxx_den, label=titles[i])
        else:
            plt.semilogy(f, Pxx_den, label=titles[i])
            plt.semilogy(f, np.abs(Pxx_den - gt_Pxx_den), '--', label=f'{titles[i]} - Ground Truth')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.legend()
    plt.title('Frequency Domain Analysis')
    plt.show()

# Edge Preservation Metrics
def edge_preservation_metrics(gt_image, predicted_image):
    gt_image_gray = rgb2gray(gt_image)
    predicted_image_gray = rgb2gray(predicted_image)
    edges_gt = filters.sobel(gt_image_gray)
    edges_pred = filters.sobel(predicted_image_gray)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(edges_gt, cmap='gray')
    plt.title('Edges Ground Truth')
    
    plt.subplot(1, 2, 2)
    plt.imshow(edges_pred, cmap='gray')
    plt.title('Edges Predicted')
    plt.show()

# Texture Analysis
def texture_analysis(gt_image, predicted_image):
    gt_image_gray = rgb2gray(gt_image)
    predicted_image_gray = rgb2gray(predicted_image)
    lbp_gt = local_binary_pattern(gt_image_gray, P=8, R=1, method='uniform')
    lbp_pred = local_binary_pattern(predicted_image_gray, P=8, R=1, method='uniform')
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(lbp_gt, cmap='gray')
    plt.title('LBP Ground Truth')
    
    plt.subplot(1, 2, 2)
    plt.imshow(lbp_pred, cmap='gray')
    plt.title('LBP Predicted')
    plt.show()

# Structural Similarity at Multiple Scales
def ms_ssim(gt_image, predicted_image):
    ms_ssim_value = ssim(gt_image, predicted_image, data_range=gt_image.max() - gt_image.min(), multichannel=True)
    print(f'MS-SSIM: {ms_ssim_value}')

# Information Fidelity Criterion (IFC) and Visual Information Fidelity (VIF)
def ifc_vif(gt_image, predicted_image):
    ifc_value = peak_signal_noise_ratio(gt_image, predicted_image)
    vif_value = mean_squared_error(gt_image, predicted_image)
    print(f'IFC: {ifc_value}, VIF: {vif_value}')

def main():
    # Load Data
    image_folder = 'DIV2K_valid_HR.nosync'
    train_noise_levels = [10, 20, 30, 40, 50, 60, 70, 80]
    val_noise_levels = [10, 20, 30, 40, 50, 60, 70, 80]

    train_loader, val_loader = load_data(image_folder, batch_size=1, num_workers=8, validation_split=0.5, augment=False, dataset_percentage=0.01, only_validation=False, include_noise_level=True, train_noise_levels=train_noise_levels, val_noise_levels=val_noise_levels, use_rgb=True)

    # Load Models
    unet_model_path = "checkpoints/unet_denoising.pth"
    diffusion_model_path = "checkpoints/diffusion_model_checkpointed_epoch_200.pth"

    unet_model = UNet().to(device)
    unet_checkpoint = torch.load(unet_model_path, map_location=device)
    if 'model_state_dict' in unet_checkpoint:
        unet_model.load_state_dict(unet_checkpoint['model_state_dict'])
    else:
        unet_model.load_state_dict(unet_checkpoint)
    unet_model.eval()

    diffusion_model = DiffusionModel(UNet_S_Checkpointed()).to(device)
    diffusion_checkpoint = torch.load(diffusion_model_path, map_location=device)
    if isinstance(diffusion_checkpoint, dict) and 'model_state_dict' in diffusion_checkpoint:
        diffusion_model.load_state_dict(diffusion_checkpoint['model_state_dict'])
    else:
        diffusion_model = diffusion_checkpoint
    diffusion_model.eval()

    # Initialize LPIPS and DISTS models
    lpips_model = lpips.LPIPS(net='alex').to(device)
    dists_model = DISTS().to(device)

    # Evaluation
    for data in val_loader:
        degraded_image, gt_image, noise_level = data

        degraded_image = degraded_image.to(device)
        gt_image = gt_image.to(device)

        with torch.no_grad():
            t = torch.randint(0, diffusion_model.timesteps, (1,), device=device).float() / diffusion_model.timesteps
            predicted_diffusion = diffusion_model.improved_sampling(degraded_image)

            degraded_image_1 = degraded_image.mean(dim=1, keepdim=True)
            gt_image_1 = gt_image.mean(dim=1, keepdim=True)

            predicted_unet = unet_model(degraded_image_1)

        for j in range(degraded_image.size(0)):
            # Extract images for analysis
            gt_img_np = gt_image[j].cpu().numpy().squeeze()
            degraded_img_np = degraded_image[j].cpu().numpy().squeeze()
            predicted_diffusion_np = predicted_diffusion[j].cpu().numpy().squeeze()
            predicted_unet_np = predicted_unet[j].cpu().numpy().squeeze()

            # Convert to RGB if single-channel
            if gt_img_np.ndim == 2:
                gt_img_np = np.stack([gt_img_np] * 3, axis=-1)
                degraded_img_np = np.stack([degraded_img_np] * 3, axis=-1)
                predicted_diffusion_np = np.stack([predicted_diffusion_np] * 3, axis=-1)
                predicted_unet_np = np.stack([predicted_unet_np] * 3, axis=-1)

            # Frequency Domain Analysis
            frequency_domain_analysis([gt_img_np, degraded_img_np, predicted_unet_np, predicted_diffusion_np],
                                      ['Ground Truth', 'Noisy', 'UNet', 'Diffusion'], gt_img_np)

            # Edge Preservation Metrics
            edge_preservation_metrics(gt_img_np, predicted_unet_np)
            edge_preservation_metrics(gt_img_np, predicted_diffusion_np)

            # Texture Analysis
            texture_analysis(gt_img_np, predicted_unet_np)
            texture_analysis(gt_img_np, predicted_diffusion_np)

            # MS-SSIM
            ms_ssim(gt_img_np, predicted_unet_np)
            ms_ssim(gt_img_np, predicted_diffusion_np)

            # IFC and VIF
            ifc_vif(gt_img_np, predicted_unet_np)
            ifc_vif(gt_img_np, predicted_diffusion_np)

if __name__ == "__main__":
    main()
