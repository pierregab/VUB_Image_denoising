import torch
import subprocess
import sys
import os

# Assuming your script is in RCA_GAN and the project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data
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
                                         validation_split=0.2, augment=False, dataset_percentage=0.05)

    # Train the model with the current hyperparameters
    train_rca_gan(
        train_loader, val_loader, num_epochs=100, betas=(0.5, 0.999), init_type='xavier',
    )

if __name__ == '__main__':
    main()