import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
    psnr_value = psnr(original_np, processed_np)
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

    # Predict and plot results for specified number of images from the validation set
    fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    image_count = 0

    for degraded_image, gt_image in val_loader:
        degraded_image = degraded_image.to(device)
        gt_image = gt_image.to(device)

        with torch.no_grad():
            predicted_image = model(degraded_image)

        for j in range(degraded_image.size(0)):
            if image_count >= num_images:
                break

            degraded_np = degraded_image[j].cpu().squeeze().numpy()
            predicted_np = predicted_image[j].cpu().squeeze().numpy()
            gt_np = gt_image[j].cpu().squeeze().numpy()

            psnr_degraded, ssim_degraded = compute_metrics(gt_image[j], degraded_image[j])
            psnr_predicted, ssim_predicted = compute_metrics(gt_image[j], predicted_image[j])

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

            if image_count >= num_images:
                break

    plt.tight_layout()
    plt.show()
