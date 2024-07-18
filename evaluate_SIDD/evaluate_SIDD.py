import os
import numpy as np
import scipy.io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from diffusion_denoising.diffusion_RDUnet import RDUNet_T, DiffusionModel  # Ensure your diffusion model script is correctly imported

class SIDDMatDataset(Dataset):
    def __init__(self, noisy_mat_file, gt_mat_file):
        self.noisy_data = scipy.io.loadmat(noisy_mat_file)['ValidationNoisyBlocksSrgb']
        self.gt_data = scipy.io.loadmat(gt_mat_file)['ValidationGtBlocksSrgb']

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return self.noisy_data.shape[0] * self.noisy_data.shape[1]

    def __getitem__(self, idx):
        img_idx = idx // self.noisy_data.shape[1]
        patch_idx = idx % self.noisy_data.shape[1]

        noisy_patch = self.noisy_data[img_idx, patch_idx]
        gt_patch = self.gt_data[img_idx, patch_idx]

        noisy_patch = self.transform(noisy_patch)
        gt_patch = self.transform(gt_patch)

        return noisy_patch, gt_patch

def evaluate_model(model, dataloader, device):
    psnr_values = []
    ssim_values = []
    inference_times = []
    sample_images = []

    model.eval()
    with torch.no_grad():
        for i, (noisy, gt) in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            noisy = noisy.to(device)
            gt = gt.to(device)

            start_time = time.time()
            denoised = model.improved_sampling(noisy)  # Adjusted for the diffusion model
            inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            output = denoised.cpu().numpy().squeeze().transpose(1, 2, 0)  # Convert from CHW to HWC format
            gt = gt.cpu().numpy().squeeze().transpose(1, 2, 0)  # Convert from CHW to HWC format
            noisy = noisy.cpu().numpy().squeeze().transpose(1, 2, 0)  # Convert from CHW to HWC format

            psnr_value = peak_signal_noise_ratio(gt, output, data_range=1)
            ssim_value = structural_similarity(gt, output, data_range=1, multichannel=True, channel_axis=-1)

            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            inference_times.append(inference_time)

            # Store some sample images for debugging
            if i > 10 and i < 15:
                sample_images.append((noisy, gt, output))

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_inference_time = np.mean(inference_times)

    return avg_psnr, avg_ssim, avg_inference_time, sample_images

def plot_sample_images(sample_images):
    fig, axs = plt.subplots(len(sample_images), 3, figsize=(15, 5 * len(sample_images)))
    for i, (noisy, gt, output) in enumerate(sample_images):
        axs[i, 0].imshow(noisy)
        axs[i, 0].set_title('Noisy')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(gt)
        axs[i, 1].set_title('Ground Truth')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(output)
        axs[i, 2].set_title('Denoised')
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # Paths to the dataset files
    noisy_mat_file = 'evaluate_SIDD/ValidationNoisyBlocksSrgb.mat'
    gt_mat_file = 'evaluate_SIDD/ValidationGtBlocksSrgb.mat'

    # Initialize dataset and dataloader
    dataset = SIDDMatDataset(noisy_mat_file, gt_mat_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Load your model
    base_filters = 32
    timesteps = 20
    model = DiffusionModel(RDUNet_T(base_filters=base_filters), timesteps=timesteps)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model.to(device)

    # Load the model checkpoint
    checkpoint_path = 'checkpoints/diffusion_RDUnet_model_checkpointed_epoch_34.pth'  # Adjust path as needed
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])  # Only load the model parameters

    # Evaluate the model
    avg_psnr, avg_ssim, avg_inference_time, sample_images = evaluate_model(model, dataloader, device)

    # Print the results
    print(f'Average PSNR: {avg_psnr:.2f}')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Average Inference Time: {avg_inference_time:.2f} ms')

    # Save the results
    results = {
        'Method': ['YourModel'],  # Replace with your method name
        'MACs (G)': ['Your MACs'],  # Replace with your model MACs
        'Inference Time (ms)': [avg_inference_time],
        'PSNR': [avg_psnr],
        'SSIM': [avg_ssim]
    }
    
    df = pd.DataFrame(results)
    df.to_csv('benchmark_results.csv', index=False)

    # Plot sample images for debugging
    plot_sample_images(sample_images)

if __name__ == '__main__':
    main()
