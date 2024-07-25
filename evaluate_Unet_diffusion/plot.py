import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import ImageGrid

def save_example_images(example_images, save_dir):
    noise_levels_to_plot = [10, 30, 50]
    filtered_images = {k: v for k, v in example_images.items() if k in noise_levels_to_plot}
    num_levels = len(filtered_images)
    
    if num_levels == 0:
        print("No example images to plot.")
        return
    
    fig = plt.figure(figsize=(16, 5 * num_levels))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(num_levels, 4),
                     axes_pad=0.6,
                     share_all=True,
                     )
    
    # Function to process and prepare image for plotting
    def process_image(img):
        img_np = np.array(img)
        if img_np.ndim == 3 and img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        return img_np
    
    vmin, vmax = 0, 255
    for i, (sigma, images) in enumerate(filtered_images.items()):
        gt_image, degraded_image, predicted_unet_image, predicted_diffusion_image = images
        
        images_to_plot = [
            ("Ground Truth", process_image(gt_image)),
            ("Noisy", process_image(degraded_image)),
            ("Denoised (UNet)", process_image(predicted_unet_image)),
            ("Denoised (Diffusion)", process_image(predicted_diffusion_image))
        ]
        
        for j, (title, img) in enumerate(images_to_plot):
            ax = grid[i*4 + j]
            im = ax.imshow(img, cmap='gray' if img.ndim == 2 else None, vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}\n(σ = {sigma})", fontsize=12, fontweight='bold')
            ax.axis('off')
            
    
    plt.suptitle("Image Denoising Comparison Across Noise Levels", fontsize=16, fontweight='bold', y=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'example_images_comparison.png'), dpi=300, bbox_inches='tight')
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

    # Ensure axs is a 2D array even if num_levels is 1
    if num_levels == 1:
        axs = np.array([axs])

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
    sem_mae_diff_unet = []
    sem_mae_diff_diffusion = []

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
        sem_mae_diff_unet.append(np.std(mae_diff_unet) / np.sqrt(len(mae_diff_unet)))
        sem_mae_diff_diffusion.append(np.std(mae_diff_diffusion) / np.sqrt(len(mae_diff_diffusion)))

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot with error bands
    ax.plot(unique_noise_levels, avg_mae_diff_unet, '-', label='UNet Model', color='#7FDBFF', linewidth=2.5, marker='o', markersize=8)
    ax.fill_between(unique_noise_levels, 
                    np.array(avg_mae_diff_unet) - np.array(sem_mae_diff_unet),
                    np.array(avg_mae_diff_unet) + np.array(sem_mae_diff_unet),
                    color='#7FDBFF', alpha=0.2)

    ax.plot(unique_noise_levels, avg_mae_diff_diffusion, '-', label='Diffusion Model', color='#2ECC40', linewidth=2.5, marker='s', markersize=8)
    ax.fill_between(unique_noise_levels, 
                    np.array(avg_mae_diff_diffusion) - np.array(sem_mae_diff_diffusion),
                    np.array(avg_mae_diff_diffusion) + np.array(sem_mae_diff_diffusion),
                    color='#2ECC40', alpha=0.2)

    ax.set_xlabel('Noise Standard Deviation (σ)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAE in High-Frequency Domain', fontsize=14, fontweight='bold')
    ax.set_title('High-Frequency Domain Analysis of Denoising Models', fontsize=16, fontweight='bold')

    ax.legend(fontsize=12, loc='upper right', frameon=True, facecolor='white', edgecolor='none', shadow=True)
    ax.grid(True, which="both", ls="--", alpha=0.3, color='gray')

    # Format ticks
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Set y-axis to log scale
    ax.set_yscale('log')

    # Add textbox with parameters
    textstr = f'High Freq. Threshold: {high_freq_threshold}\nEpoch: {last_epoch}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'high_frequency_domain_analysis.png'), dpi=300, bbox_inches='tight')
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
    
    # Create a colorblind-friendly colormap
    colors = ['#FFEDA0', '#FEB24C', '#F03B20']  # Light yellow to orange to dark red
    n_bins = len(unique_noise_levels)
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    for nl in unique_noise_levels:
        idx = (noise_levels == nl) & (np.array(metrics['epoch']) == last_epoch)
        psd_gt_all, psd_unet_all, psd_diffusion_all, psd_degraded_all = [], [], [], []
        
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

        # Calculate standard error
        se_gt = np.std(psd_gt_all, axis=0) / np.sqrt(len(psd_gt_all))
        se_degraded = np.std(psd_degraded_all, axis=0) / np.sqrt(len(psd_degraded_all))
        se_unet = np.std(psd_unet_all, axis=0) / np.sqrt(len(psd_unet_all))
        se_diffusion = np.std(psd_diffusion_all, axis=0) / np.sqrt(len(psd_diffusion_all))

        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot with error bands
        ax.plot(f_gt[high_freq_idx], avg_psd_gt, label='Ground Truth', color='#000000', linewidth=2.5, zorder=5)
        ax.fill_between(f_gt[high_freq_idx], avg_psd_gt - se_gt, avg_psd_gt + se_gt, color='#000000', alpha=0.1, zorder=4)
        
        ax.plot(f_gt[high_freq_idx], avg_psd_degraded, label='Degraded', color='#FF4136', linewidth=2.5, zorder=3)
        ax.fill_between(f_gt[high_freq_idx], avg_psd_degraded - se_degraded, avg_psd_degraded + se_degraded, color='#FF4136', alpha=0.1, zorder=2)
        
        ax.plot(f_gt[high_freq_idx], avg_psd_unet, label='UNet Model', color='#7FDBFF', linewidth=2.5, zorder=3)
        ax.fill_between(f_gt[high_freq_idx], avg_psd_unet - se_unet, avg_psd_unet + se_unet, color='#7FDBFF', alpha=0.2, zorder=2)
        
        ax.plot(f_gt[high_freq_idx], avg_psd_diffusion, label=f'Diffusion Model', color='#2ECC40', linewidth=2.5, zorder=3)
        ax.fill_between(f_gt[high_freq_idx], avg_psd_diffusion - se_diffusion, avg_psd_diffusion + se_diffusion, color='#2ECC40', alpha=0.1, zorder=2)

        ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Power Spectral Density (dB/Hz)', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(f'Power Spectral Density Comparison\nNoise Level σ = {nl:.2f}', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='lower left', frameon=True, facecolor='white', edgecolor='none', shadow=True)
        ax.grid(True, which="both", ls="--", alpha=0.3, color='gray')
        
        # Format ticks
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add textbox with parameters
        textstr = f'High Freq. Threshold: {high_freq_threshold}\nEpoch: {last_epoch}\nNoise Level (σ): {nl:.2f}'
        props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'psd_comparison_noise_level_{nl:.2f}.png'), dpi=300, bbox_inches='tight')
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

def generate_comparison_plot(metrics, epochs, save_dir, use_bm3d=False):
    noise_levels = np.array(metrics['noise_level'])
    unique_noise_levels = sorted(np.unique(noise_levels))
    
    psnr_diffusion = np.array(metrics['psnr_diffusion'])
    psnr_unet = np.array(metrics['psnr_unet'])
    lpips_diffusion = np.array(metrics['lpips_diffusion'])
    lpips_unet = np.array(metrics['lpips_unet'])
    
    # Check if BM3D data exists and if use_bm3d is True
    use_bm3d = use_bm3d and 'psnr_bm3d' in metrics and 'lpips_bm3d' in metrics
    if use_bm3d:
        psnr_bm3d = np.array(metrics['psnr_bm3d'])
        lpips_bm3d = np.array(metrics['lpips_bm3d'])

    avg_psnr_diffusion = [np.mean(psnr_diffusion[noise_levels == nl]) for nl in unique_noise_levels]
    avg_psnr_unet = [np.mean(psnr_unet[noise_levels == nl]) for nl in unique_noise_levels]
    avg_lpips_diffusion = [np.mean(lpips_diffusion[noise_levels == nl]) for nl in unique_noise_levels]
    avg_lpips_unet = [np.mean(lpips_unet[noise_levels == nl]) for nl in unique_noise_levels]
    
    if use_bm3d:
        avg_psnr_bm3d = [np.mean(psnr_bm3d[noise_levels == nl]) for nl in unique_noise_levels]
        avg_lpips_bm3d = [np.mean(lpips_bm3d[noise_levels == nl]) for nl in unique_noise_levels]

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a colorblind-friendly colormap
    colors = ['#FFEDA0', '#FEB24C', '#F03B20']  # Light yellow to orange to dark red
    n_bins = len(unique_noise_levels)
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # Plot with color gradient
    scatter_diffusion = ax.scatter(avg_lpips_diffusion, avg_psnr_diffusion,
                                   c=unique_noise_levels, cmap=cmap,
                                   label='Diffusion Model', marker='*', s=200, edgecolors='black')
    
    scatter_unet = ax.scatter(avg_lpips_unet, avg_psnr_unet,
                              c=unique_noise_levels, cmap=cmap,
                              label='UNet Model', marker='o', s=200, edgecolors='black')
    
    if use_bm3d:
        scatter_bm3d = ax.scatter(avg_lpips_bm3d, avg_psnr_bm3d,
                                  c=unique_noise_levels, cmap=cmap,
                                  label='BM3D', marker='^', s=200, edgecolors='black')
    
    # Add colorbar
    cbar = plt.colorbar(scatter_diffusion)
    cbar.set_label('Noise Level', rotation=270, labelpad=15)
    
    ax.set_xlabel('LPIPS (lower is better)', fontsize=12)
    ax.set_ylabel('PSNR (higher is better)', fontsize=12)
    ax.set_title('Model Comparison across Noise Levels', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add arrows to indicate better performance directions
    ax.annotate('', xy=(0.05, 0.95), xytext=(0.15, 0.95),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.text(0.1, 0.97, 'Better LPIPS', ha='center', va='center',
            transform=ax.transAxes, fontsize=10, color='gray')

    ax.annotate('', xy=(0.95, 0.85), xytext=(0.95, 0.95),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='<-', color='gray'))
    ax.text(0.97, 0.9, 'Better PSNR', ha='center', va='center',
            transform=ax.transAxes, fontsize=10, color='gray', rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_plot.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)