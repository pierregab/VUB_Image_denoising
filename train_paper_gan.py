import torch
import subprocess
from data_loader import load_data
from paper_gan import train_rca_gan

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
    log_dir = 'runs/paper_gan'
    start_tensorboard(log_dir)

    # Use num_workers=0 to avoid multiprocessing issues for debugging
    train_loader, val_loader = load_data(gt_folder, degraded_folder, batch_size=1, num_workers=8, 
                                         validation_split=0.2, augment=False, dataset_percentage=0.025)

    # Train the model with the current hyperparameters
    train_rca_gan(
        train_loader, val_loader, num_epochs=200,
        lambda_pixel=100, lambda_perceptual=0.1,
        lambda_texture=1.0, lr=0.0001, betas=(0.5, 0.999),
    )

if __name__ == '__main__':
    main()
