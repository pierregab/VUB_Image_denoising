import optuna
import torch
import argparse
import os
from torchvision.transforms import ToTensor
from PIL import Image
from diffusion_RDUnet import train, DiffusionModel, RDUNet_T, denormalize, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr

def extract_patch(img, top, left, patch_size=256):
    return img.crop((left, top, left + patch_size, top + patch_size))

def load_validation_patches(patch_dir, patch_size=256, num_patches=12):
    patches = []
    for img_file in sorted(os.listdir(patch_dir))[:num_patches]:  # Ensure we always use the same 12 patches
        if img_file.endswith('.png'):
            img_path = os.path.join(patch_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            width, height = img.size

            # Ensure the coordinates are consistent for each image
            top = height // 4
            left = width // 4
            patch = extract_patch(img, top, left, patch_size)
            patch = ToTensor()(patch).unsqueeze(0)  # Add batch dimension
            patches.append(patch)
    patches = torch.cat(patches, dim=0)
    return patches

def evaluate_model(model_path, patch_dir, base_filters, timesteps):
    # Load model
    unet_checkpointed = RDUNet_T(base_filters=base_filters).to(device)
    model_checkpointed = DiffusionModel(unet_checkpointed, timesteps=timesteps).to(device)
    model_checkpointed.load_state_dict(torch.load(model_path, map_location=device))
    model_checkpointed.eval()

    # Load validation patches
    validation_patches = load_validation_patches(patch_dir)

    # Perform denoising and calculate PSNR
    psnr_values = []
    with torch.no_grad():
        for patch in validation_patches:
            noisy_patch = patch.to(device)
            clean_patch = patch.to(device)  # In a real scenario, you should have the corresponding clean patches
            denoised_patch = model_checkpointed.improved_sampling(noisy_patch)
            denoised_patch = denormalize(denoised_patch.cpu())
            clean_patch = denormalize(clean_patch.cpu())
            psnr = calculate_psnr(denoised_patch, clean_patch)
            psnr_values.append(psnr)
    
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

    # Call the train function with the suggested hyperparameters and get validation loss
    train(args)

    # Evaluate model using PSNR on validation patches
    model_path = os.path.join(args.output_dir, "diffusion_RDUNet_model_checkpointed_final.pth")
    patch_dir = 'dataset/DIV2K_valid_HR.nosync'
    avg_psnr = evaluate_model(model_path, patch_dir, args.base_filters, args.timesteps)

    return -avg_psnr  # Optuna minimizes the objective, so return negative PSNR

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print(f"Best trial: {study.best_trial.value}")
    print("Best hyperparameters: ")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
