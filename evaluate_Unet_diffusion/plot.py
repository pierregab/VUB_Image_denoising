import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import ImageGrid

# Use the same palette for all plots (pale colors)
pale_red = '#FF4136'
pale_blue = '#0074D9'
pale_green = '#2ECC40'
pale_yellow = '#FFDC00'
pale_purple = '#B10DC9'

# Enable LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def save_example_images(example_images, save_dir):
    noise_levels_to_plot = [10, 30, 50]
    filtered_images = {k: v for k, v in example_images.items() if k in noise_levels_to_plot}
    num_levels = len(filtered_images)
    
    if num_levels == 0:
        print("No example images to plot.")
        return
    
    fig = plt.figure(figsize=(16, 5 * num_levels), constrained_layout=True)
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
            (r"Ground Truth", process_image(gt_image)),
            (r"Noisy", process_image(degraded_image)),
            (r"Denoised (UNet)", process_image(predicted_unet_image)),
            (r"Denoised (Diffusion)", process_image(predicted_diffusion_image))
        ]
        
        for j, (title, img) in enumerate(images_to_plot):
            ax = grid[i*4 + j]
            im = ax.imshow(img, cmap='gray' if img.ndim == 2 else None, vmin=vmin, vmax=vmax)
            ax.set_title(rf"{title} ($\sigma = {sigma}$)", fontsize=12, fontweight='bold')
            ax.axis('off')
         
    plt.suptitle(r"Image Denoising Comparison Across Noise Levels", fontsize=16, fontweight='bold', y=0.85)
    
    plt.savefig(os.path.join(save_dir, 'example_images_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_error_map(gt_image, predicted_image, save_dir):
    error_map = np.abs(gt_image - predicted_image)
    plt.imshow(error_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(r'Error Map', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(save_dir, 'error_map.png'))
    plt.close()

def save_histograms_of_differences(example_images, last_epoch, save_dir):
    noise_levels_to_plot = [15, 30, 50]
    filtered_images = {k: v for k, v in example_images.items() if k[1] in noise_levels_to_plot and k[0] == last_epoch}

    num_levels = len(filtered_images)
    if num_levels == 0:
        print("No example images to plot.")
        return

    fig, axs = plt.subplots(num_levels, 2, figsize=(20, 5 * num_levels), constrained_layout=True)

    # Ensure axs is a 2D array even if num_levels is 1
    if num_levels == 1:
        axs = np.array([axs])

    for i, ((epoch, sigma), images) in enumerate(filtered_images.items()):
        gt_image, degraded_image, predicted_unet_image, predicted_diffusion_image = images

        differences_unet = (gt_image - predicted_unet_image).flatten()
        differences_diffusion = (gt_image - predicted_diffusion_image).flatten()

        axs[i, 0].hist(differences_unet, bins=50, color=pale_blue, alpha=0.7)
        axs[i, 0].set_title(rf'Histogram of Differences (UNet) - Epoch: {epoch}, $\sigma$: {sigma}', fontsize=14, fontweight='bold')
        axs[i, 0].set_xlabel(r'Difference', fontsize=12)
        axs[i, 0].set_ylabel(r'Frequency', fontsize=12)

        axs[i, 1].hist(differences_diffusion, bins=50, color=pale_green, alpha=0.7)
        axs[i, 1].set_title(rf'Histogram of Differences (Diffusion) - Epoch: {epoch}, $\sigma$: {sigma}', fontsize=14, fontweight='bold')
        axs[i, 1].set_xlabel(r'Difference', fontsize=12)
        axs[i, 1].set_ylabel(r'Frequency', fontsize=12)

    plt.savefig(os.path.join(save_dir, 'histograms_of_differences.png'), dpi=300)
    plt.close()

def save_heatmaps(aggregated_diff_map_unet, aggregated_diff_map_diffusion, save_dir):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)

    if aggregated_diff_map_unet.ndim == 3:
        aggregated_diff_map_unet = np.mean(aggregated_diff_map_unet, axis=0)
    if aggregated_diff_map_diffusion.ndim == 3:
        aggregated_diff_map_diffusion = np.mean(aggregated_diff_map_diffusion, axis=0)

    vmin = min(aggregated_diff_map_unet.min(), aggregated_diff_map_diffusion.min())
    vmax = max(aggregated_diff_map_unet.max(), aggregated_diff_map_diffusion.max())

    im_unet = axs[0].imshow(aggregated_diff_map_unet, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    axs[0].set_title(r'Aggregated Difference Map (UNet)', fontsize=14, fontweight='bold')
    fig.colorbar(im_unet, ax=axs[0], orientation='vertical')

    im_diffusion = axs[1].imshow(aggregated_diff_map_diffusion, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    axs[1].set_title(r'Aggregated Difference Map (Diffusion)', fontsize=14, fontweight='bold')
    fig.colorbar(im_diffusion, ax=axs[1], orientation='vertical')

    plt.savefig(os.path.join(save_dir, 'heatmaps.png'), dpi=300)
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

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    # Plot with error bands
    ax.plot(unique_noise_levels, avg_mae_diff_unet, '-', label=r'UNet Model', color=pale_blue, linewidth=2.5, marker='o', markersize=8)
    ax.fill_between(unique_noise_levels, 
                    np.array(avg_mae_diff_unet) - np.array(sem_mae_diff_unet),
                    np.array(avg_mae_diff_unet) + np.array(sem_mae_diff_unet),
                    color=pale_blue, alpha=0.2)

    ax.plot(unique_noise_levels, avg_mae_diff_diffusion, '-', label=r'Diffusion Model', color=pale_green, linewidth=2.5, marker='s', markersize=8)
    ax.fill_between(unique_noise_levels, 
                    np.array(avg_mae_diff_diffusion) - np.array(sem_mae_diff_diffusion),
                    np.array(avg_mae_diff_diffusion) + np.array(sem_mae_diff_diffusion),
                    color=pale_green, alpha=0.2)

    ax.set_xlabel(r'Noise Standard Deviation ($\sigma$)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'MAE in High-Frequency Domain', fontsize=14, fontweight='bold')
    ax.set_title(r'High-Frequency Domain Analysis of Denoising Models', fontsize=16, fontweight='bold')

    ax.legend(fontsize=12, loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, which="both", ls="--", alpha=0.3, color='gray')

    # Format ticks
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Set y-axis to log scale
    ax.set_yscale('log')

    # Add textbox with parameters
    textstr = rf'High Freq. Threshold: {high_freq_threshold}\\ Epoch: {last_epoch}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

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

    plt.figure(figsize=(10, 6), constrained_layout=True)
    plt.plot(unique_noise_levels, avg_mae_diff_unet, 'o-', label=r'UNet Model', color=pale_purple)
    colors = [pale_green, pale_blue, pale_red, pale_yellow, 'black', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    for idx, epoch in enumerate(epochs):
        plt.plot(unique_noise_levels, avg_mae_diff_diffusion[epoch], 'o-', label=rf'Diffusion Model (Epoch {epoch})', color=colors[idx % len(colors)])
    plt.xlabel(r'Noise Standard Deviation ($\sigma$)', fontsize=14)
    plt.ylabel(r'MAE in High-Frequency Domain', fontsize=14)
    plt.title(r'MAE in High-Frequency Domain Analysis', fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'high_frequency_domain_analysis_multiple_epochs.png'))
    plt.close()

def plot_psd_comparison(metrics, last_epoch, save_dir, high_freq_threshold=0.5):
    noise_levels = np.array(metrics['noise_level'])
    unique_noise_levels = sorted(np.unique(noise_levels))
    
    # Create a colorblind-friendly colormap
    colors = [pale_yellow, pale_red, pale_green]  # Light yellow to orange to dark red
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

        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
        
        # Plot with error bands
        ax.plot(f_gt[high_freq_idx], avg_psd_gt, label='Ground Truth', color='#000000', linewidth=2.5, zorder=5)
        ax.fill_between(f_gt[high_freq_idx], avg_psd_gt - se_gt, avg_psd_gt + se_gt, color='#000000', alpha=0.1, zorder=4)
        
        ax.plot(f_gt[high_freq_idx], avg_psd_degraded, label='Degraded', color='#FF4136', linewidth=2.5, zorder=3)
        ax.fill_between(f_gt[high_freq_idx], avg_psd_degraded - se_degraded, avg_psd_degraded + se_degraded, color='#FF4136', alpha=0.1, zorder=2)
        
        ax.plot(f_gt[high_freq_idx], avg_psd_unet, label='UNet Model', color='#7FDBFF', linewidth=2.5, zorder=3)
        ax.fill_between(f_gt[high_freq_idx], avg_psd_unet - se_unet, avg_psd_unet + se_unet, color='#7FDBFF', alpha=0.2, zorder=2)
        
        ax.plot(f_gt[high_freq_idx], avg_psd_diffusion, label=f'Diffusion Model', color='#2ECC40', linewidth=2.5, zorder=3)
        ax.fill_between(f_gt[high_freq_idx], avg_psd_diffusion - se_diffusion, avg_psd_diffusion + se_diffusion, color='#2ECC40', alpha=0.1, zorder=2)

        ax.set_xlabel(r'Frequency (Hz)', fontsize=14, fontweight='bold')
        ax.set_ylabel(r'Power Spectral Density (dB/Hz)', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(rf'Power Spectral Density Comparison, Noise Level $\sigma$ = {nl:.2f}', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='lower left', frameon=True, facecolor='white', edgecolor='none')
        ax.grid(True, which="both", ls="--", alpha=0.3, color='gray')
        
        # Format ticks
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add textbox with parameters
        textstr = rf'High Freq. Threshold: {high_freq_threshold}\\ Epoch: {last_epoch}\\ Noise Level ($\sigma$): {nl:.2f}'
        props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.savefig(os.path.join(save_dir, f'psd_comparison_noise_level_{nl:.2f}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def save_dists(metrics, last_epoch, save_dir):
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

    fig, axs = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

    axs.plot(unique_noise_levels, avg_dists_degraded, 'o-', label=r'Degraded', color=pale_red)
    axs.plot(unique_noise_levels, avg_dists_unet, 'o-', label=r'UNet Model', color=pale_purple)
    axs.plot(unique_noise_levels, avg_dists_diffusion_last, 'o-', label=rf'Diffusion Model (Epoch {last_epoch})', color=pale_green)
    axs.plot(unique_noise_levels, avg_dists_bm3d, 'o-', label=r'BM3D', color=pale_blue)

    axs.set_xlabel(r'Noise Standard Deviation ($\sigma$)', fontsize=14, fontweight='bold')
    axs.set_ylabel(r'DISTS', fontsize=14, fontweight='bold')
    axs.set_title(r'DISTS Value Variation', fontsize=16, fontweight='bold')
    axs.legend(fontsize=12)
    axs.grid(True, which="both", ls="--", alpha=0.3)

    plt.savefig(os.path.join(save_dir, 'dists.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_inference_time_plot(inference_times, save_dir):
    avg_inference_time_unet = np.mean(inference_times['unet'])
    avg_inference_time_diffusion = np.mean(inference_times['diffusion'])

    labels = [r'UNet', r'Diffusion']
    times = [avg_inference_time_unet, avg_inference_time_diffusion]

    plt.figure(figsize=(10, 6), constrained_layout=True)
    plt.bar(labels, times, color=[pale_purple, pale_green])
    plt.ylabel(r'Average Inference Time (s)', fontsize=14, fontweight='bold')
    plt.title(r'Average Inference Time Comparison', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(save_dir, 'inference_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_plot(metrics, epochs, save_dir, use_bm3d=False):
    noise_levels = np.array(metrics['noise_level'])
    unique_noise_levels = sorted(np.unique(noise_levels))
    
    psnr_diffusion = np.array(metrics['psnr_diffusion'])
    psnr_unet = np.array(metrics['psnr_unet'])
    lpips_diffusion = np.array(metrics['lpips_diffusion'])
    lpips_unet = np.array(metrics['lpips_unet'])
    
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

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    
    # Create a colorblind-friendly colormap
    colors = ['#FFEDA0', '#FEB24C', '#F03B20']  # Light yellow to orange to dark red
    n_bins = len(unique_noise_levels)
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    

    # Plot with color gradient
    scatter_diffusion = ax.scatter(avg_lpips_diffusion, avg_psnr_diffusion,
                                   c=unique_noise_levels, cmap=cmap,
                                   label=r'Diffusion Model', marker='*', s=200, edgecolors='black')
    
    scatter_unet = ax.scatter(avg_lpips_unet, avg_psnr_unet,
                              c=unique_noise_levels, cmap=cmap,
                              label=r'UNet Model', marker='o', s=200, edgecolors='black')
    
    if use_bm3d:
        scatter_bm3d = ax.scatter(avg_lpips_bm3d, avg_psnr_bm3d,
                                  c=unique_noise_levels, cmap=cmap,
                                  label=r'BM3D', marker='^', s=200, edgecolors='black')
    
    cbar = plt.colorbar(scatter_diffusion)
    cbar.set_label(r'Noise Level ($\sigma$)', rotation=270, labelpad=15)
    
    ax.set_xlabel(r'LPIPS (lower is better)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'PSNR (higher is better)', fontsize=14, fontweight='bold')
    ax.set_title(r'Model Comparison Across Noise Levels', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add arrows to indicate better performance directions
    ax.annotate('', xy=(0.05, 0.95), xytext=(0.15, 0.95),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.text(0.1, 0.97, r'Better LPIPS', ha='center', va='center',
            transform=ax.transAxes, fontsize=10, color='gray')

    ax.annotate('', xy=(0.95, 0.85), xytext=(0.95, 0.95),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='<-', color='gray'))
    ax.text(0.97, 0.9, 'Better PSNR', ha='center', va='center',
            transform=ax.transAxes, fontsize=10, color='gray', rotation=90)

    plt.savefig(os.path.join(save_dir, 'comparison_plot.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

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
