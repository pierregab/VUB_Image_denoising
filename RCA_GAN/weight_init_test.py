import torch
import torchvision
import subprocess
import sys
import os

# Assuming your script is in RCA_GAN and the project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset_creation.data_loader import load_data
from paper_gan import train_rca_gan, Generator, denormalize

def save_images(images, output_dir, epoch, image_type):
    for i, img in enumerate(images):
        img_path = os.path.join(output_dir, f"{image_type}_epoch_{epoch}_img_{i}.png")
        torchvision.utils.save_image(img, img_path, normalize=True)

# Define paths to ground truth and degraded images
gt_folder = 'DIV2K_train_HR.nosync/resized_ground_truth_images'
degraded_folder = 'DIV2K_train_HR.nosync/degraded_images'

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    train_loader, val_loader = load_data(gt_folder, degraded_folder, batch_size=1, num_workers=8, 
                                         validation_split=0.2, augment=True, dataset_percentage=0.05)

    # Train the model with the current hyperparameters
    initialization_types = ['normal', 'xavier', 'kaiming']
    for init_type in initialization_types:
        print(f"Training with {init_type} initialization")
        current_output_dir = f"output_images/{init_type}"
        os.makedirs(current_output_dir, exist_ok=True)
        
        # Train the model
        generator, discriminator = train_rca_gan(
            train_loader, val_loader, num_epochs=1, lr=0.0001, betas=(0.5, 0.999), init_type=init_type, log_dir=current_output_dir, use_tensorboard=False, device=device
        )
        
        # Save example images and intermediate outputs after training
        with torch.no_grad():
            for i, (degraded_images, gt_images) in enumerate(train_loader):
                if i == 0:  # Save images from the first batch
                    degraded_images = degraded_images.to(device)
                    gt_images = gt_images.to(device)
                    generator = generator.to(device)  # Ensure generator is on the correct device
                    example_degraded = degraded_images[:4].cpu()  # Take first 4 examples from the last batch
                    example_gt = gt_images[:4].cpu()
                    example_gen, intermediate_outputs = generator(degraded_images.to(device))
                    example_gen = example_gen.cpu()

                    # Denormalize images to [0, 1] for better visualization
                    example_degraded = denormalize(example_degraded)
                    example_gt = denormalize(example_gt)
                    example_gen = denormalize(example_gen)

                    # Save images
                    save_images(example_degraded, current_output_dir, 1, 'degraded')
                    save_images(example_gen, current_output_dir, 1, 'generated')
                    save_images(example_gt, current_output_dir, 1, 'ground_truth')
                    
                    # Save intermediate outputs
                    for name, output in intermediate_outputs.items():
                        output = output[:4].cpu()  # Take first 4 examples
                        output = denormalize(output)
                        if output.size(1) == 1:  # Convert single-channel to 3-channel
                            output = output.repeat(1, 3, 1, 1)
                        elif output.size(1) > 3:  # If more than 3 channels, take only the first 3 channels
                            output = output[:, :3, :, :]
                        save_images(output, current_output_dir, 1, name)
                    break  # We only need to save images from the first batch after training

if __name__ == '__main__':
    main()
