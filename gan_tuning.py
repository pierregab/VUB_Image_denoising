import torch
import optuna
import subprocess
from data_loader import load_data
from train_denoising_gan_hyperparameters import train_denoising_gan

# Define paths to ground truth and degraded images
gt_folder = 'DIV2K_train_HR.nosync/resized_ground_truth_images'
degraded_folder = 'DIV2K_train_HR.nosync/degraded_images'

def start_tensorboard(log_dir):
    try:
        subprocess.Popen(['tensorboard', '--logdir', log_dir])
        print(f"TensorBoard started at http://localhost:6006")
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")

def objective(trial):
    # Hyperparameter search space
    num_epochs = trial.suggest_int('num_epochs', 5, 50)
    lambda_pixel = trial.suggest_float('lambda_pixel', 1, 100, log=True)
    lambda_perceptual = trial.suggest_float('lambda_perceptual', 0.01, 1.0, log=True)
    lambda_edge = trial.suggest_float('lambda_edge', 0.1, 10, log=True)
    lambda_gp = trial.suggest_float('lambda_gp', 1, 10, log=True)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    beta1 = trial.suggest_float('beta1', 0.5, 0.9)
    beta2 = trial.suggest_float('beta2', 0.9, 0.999)

    # Load the dataset
    train_loader, val_loader = load_data(gt_folder, degraded_folder, batch_size=4, num_workers=8, validation_split=0.2)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # Train the model with the current hyperparameters
    best_val_loss = train_denoising_gan(
        train_loader, val_loader, num_epochs=num_epochs,
        lambda_pixel=lambda_pixel, lambda_perceptual=lambda_perceptual,
        lambda_edge=lambda_edge, lambda_gp=lambda_gp, lr=lr, betas=(beta1, beta2),
        device=device, log_dir='runs/denoising_gan', checkpoint_dir='checkpoints',
        checkpoint_prefix='denoising_gan', trial=trial
    )

    return best_val_loss

if __name__ == "__main__":
    log_dir = 'runs/denoising_gan'
    start_tensorboard(log_dir)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print(f"Best trial: {study.best_trial.params}")
