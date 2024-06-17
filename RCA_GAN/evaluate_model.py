import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import sys

from paper_gan import Generator

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data

# Assuming the necessary classes and functions are already defined:
# - Generator
# - compute_metrics
# - evaluate_model
# - load_data

def compute_metrics(original, processed):
    """
    Compute PSNR and SSIM between original and processed images.

    Args:
        original (torch.Tensor): Original ground truth image.
        processed (torch.Tensor): Processed image to compare.

    Returns:
        tuple: PSNR and SSIM values.
    """
    original_np = original.cpu().numpy().squeeze()
    processed_np = processed.cpu().numpy().squeeze()
    psnr_value = psnr(original_np, processed_np, data_range=processed_np.max() - processed_np.min())
    ssim_value = ssim(original_np, processed_np, data_range=processed_np.max() - processed_np.min())
    return psnr_value, ssim_value

def evaluate_model(model, val_loader, device, num_images=4, model_path="best_denoising_unet_b&w.pth"):
    """
    Evaluate a model on the validation set and compute metrics.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        num_images (int): Number of images to display for evaluation.
        model_path (str): Path to the trained model weights.

    Returns:
        None
    """
    # Load the trained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Lists to store PSNR and SSIM values
    psnr_degraded_list = []
    ssim_degraded_list = []
    psnr_predicted_list = []
    ssim_predicted_list = []

    # Predict and plot results for specified number of images from the validation set
    fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    image_count = 0

    for i, (degraded_image, gt_image) in enumerate(val_loader):
        degraded_image = degraded_image.to(device)
        gt_image = gt_image.to(device)

        with torch.no_grad():
            predicted_image, _ = model(degraded_image)

        for j in range(degraded_image.size(0)):
            psnr_degraded, ssim_degraded = compute_metrics(gt_image[j], degraded_image[j])
            psnr_predicted, ssim_predicted = compute_metrics(gt_image[j], predicted_image[j])

            psnr_degraded_list.append(psnr_degraded)
            ssim_degraded_list.append(ssim_degraded)
            psnr_predicted_list.append(psnr_predicted)
            ssim_predicted_list.append(ssim_predicted)

            if image_count < num_images:
                degraded_np = degraded_image[j].cpu().squeeze().numpy()
                predicted_np = predicted_image[j].cpu().squeeze().numpy()
                gt_np = gt_image[j].cpu().squeeze().numpy()

                axs[image_count, 0].imshow(degraded_np, cmap='gray')
                axs[image_count, 0].set_title(f'Degraded Image\nPSNR: {psnr_degraded:.2f}, SSIM: {ssim_degraded:.4f}')
                axs[image_count, 0].axis('off')

                axs[image_count, 1].imshow(predicted_np, cmap='gray')
                axs[image_count, 1].set_title(f'Predicted Image\nPSNR: {psnr_predicted:.2f}, SSIM: {ssim_predicted:.4f}')
                axs[image_count, 1].axis('off')

                axs[image_count, 2].imshow(gt_np, cmap='gray')
                axs[image_count, 2].set_title('Ground Truth Image')
                axs[image_count, 2].axis('off')

                image_count += 1

        if image_count >= num_images and i >= len(val_loader):
            break

    plt.tight_layout()
    plt.show()

    # Compute average metrics over the entire validation set
    avg_psnr_degraded = np.mean(psnr_degraded_list)
    avg_ssim_degraded = np.mean(ssim_degraded_list)
    avg_psnr_predicted = np.mean(psnr_predicted_list)
    avg_ssim_predicted = np.mean(ssim_predicted_list)

    print(f"Average PSNR of Degraded Images: {avg_psnr_degraded:.2f}")
    print(f"Average SSIM of Degraded Images: {avg_ssim_degraded:.4f}")
    print(f"Average PSNR of Predicted Images: {avg_psnr_predicted:.2f}")
    print(f"Average SSIM of Predicted Images: {avg_ssim_predicted:.4f}")

    # Plot comparison of metrics
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].bar(['Degraded', 'Predicted'], [avg_psnr_degraded, avg_psnr_predicted], color=['red', 'green'])
    ax[0].set_title('Average PSNR Comparison')
    ax[0].set_ylabel('PSNR')

    ax[1].bar(['Degraded', 'Predicted'], [avg_ssim_degraded, avg_ssim_predicted], color=['red', 'green'])
    ax[1].set_title('Average SSIM Comparison')
    ax[1].set_ylabel('SSIM')

    plt.show()

if __name__ == "__main__":
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # Load the dataset using the provided load_data function
    image_folder = 'DIV2K_train_HR.nosync'
    train_loader, val_loader = load_data(image_folder, batch_size=1, num_workers=8, validation_split=0.2, augment=False, dataset_percentage=0.001)

    # Instantiate the model
    in_channels = 1
    out_channels = 1
    generator = Generator(in_channels, out_channels)

    # Evaluate the model
    evaluate_model(generator, val_loader, device, num_images=4, model_path="runs/generator_epoch_20.pth")
