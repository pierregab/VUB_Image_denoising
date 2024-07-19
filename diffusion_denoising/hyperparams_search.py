import optuna
import torch
import argparse
import os
from diffusion_RDUnet import train, DiffusionModel, RDUNet_T, denormalize, load_data

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr

def evaluate_model(model, val_loader):
    model.eval()
    val_noisy_images, val_clean_images = next(iter(val_loader))
    val_noisy_images, val_clean_images = val_noisy_images.to(device), val_clean_images.to(device)
    with torch.no_grad():
        denoised_images = model.improved_sampling(val_noisy_images)
        denoised_images = denormalize(denoised_images.cpu())
        val_clean_images = denormalize(val_clean_images.cpu())

        psnr_values = [calculate_psnr(denoised_images[i], val_clean_images[i]) for i in range(len(denoised_images))]
        avg_psnr = sum(psnr_values) / len(psnr_values)
    
    return avg_psnr

def objective(trial):
    # Define hyperparameter search space
    args = argparse.Namespace()
    args.dataset_choice = 'SIDD'
    args.checkpoint_path = None
    args.num_epochs = 5
    args.batch_size = 8
    args.num_workers = 8
    args.validation_split = 0.2
    args.augment = True
    args.dataset_percentage = 0.1
    args.base_filters = trial.suggest_int('base_filters', 16, 64, step=16)
    args.timesteps = trial.suggest_int('timesteps', 10, 20, step=5)
    args.optimizer_choice = trial.suggest_categorical('optimizer_choice', ['adam', 'adamw'])
    args.scheduler_choice = trial.suggest_categorical('scheduler_choice', ['cosine', 'step'])
    args.output_dir = os.path.join("checkpoints", f"trial_{trial.number}")

    if args.optimizer_choice == 'adam':
        args.lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    elif args.optimizer_choice == 'adamw':
        args.lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
        args.weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)

    # Load data
    train_loader, val_loader = load_data(args)

    # Train the model
    train(args, train_loader, val_loader)

    # Evaluate model using PSNR on validation patches
    model_path = os.path.join(args.output_dir, "diffusion_RDUNet_model_checkpointed_final.pth")
    unet_checkpointed = RDUNet_T(base_filters=args.base_filters).to(device)
    model_checkpointed = DiffusionModel(unet_checkpointed, timesteps=args.timesteps).to(device)
    model_checkpointed.load_state_dict(torch.load(model_path, map_location=device))

    avg_psnr = evaluate_model(model_checkpointed, val_loader)
    return -avg_psnr  # Optuna minimizes the objective, so return negative PSNR

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print(f"Best trial: {study.best_trial.value}")
    print("Best hyperparameters: ")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
