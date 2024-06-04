import torch
import optuna
import subprocess
from data_loader import load_data
from train_denoising_gan import train_denoising_gan

# Define paths to ground truth and degraded images
gt_folder = 'DIV2K_train_HR.nosync/resized_ground_truth_images'
degraded_folder = 'DIV2K_train_HR.nosync/degraded_images'

def start_tensorboard(log_dir):
    try:
        subprocess.Popen(['tensorboard', '--logdir', log_dir])
        print(f"TensorBoard started at http://localhost:6006")
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")

def main():
    log_dir = 'runs/denoising_gan'
    start_tensorboard(log_dir)

    # Use num_workers=0 to avoid multiprocessing issues for debugging
    train_loader, val_loader = load_data(gt_folder, degraded_folder, batch_size=16, num_workers=8, validation_split=0.2)

    # Train the model with the current hyperparameters
    train_denoising_gan(
        train_loader, val_loader, num_epochs=50,
        lambda_pixel=25.874, lambda_perceptual=0.04481, lambda_edge=0.6877, lambda_gp= 9.511,
        lr=0.0002714, betas=(0.7074, 0.92944),
    )

if __name__ == '__main__':
    main()
