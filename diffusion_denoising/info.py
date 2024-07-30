import torch
from torchinfo import summary
from old_diffusion_RDUnet import RDUNet_T, DiffusionModel
import sys
import os
import time

# Assuming your script is in RCA_GAN and the project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from UNet.RDUNet_model import RDUNet

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Load UNet model
unet_model = RDUNet(base_filters=128).to(device)
unet_summary = summary(unet_model, input_size=(1, 3, 256, 256), verbose=0)

# Load Diffusion model
unet_t_model = RDUNet_T(base_filters=32).to(device)
diffusion_model = DiffusionModel(unet_t_model).to(device)
diffusion_summary = summary(diffusion_model, input_size=[(1, 3, 256, 256), (1, 3, 256, 256), (1,)], verbose=0)

# Define a dummy input tensor
input_tensor = torch.randn(1, 3, 256, 256).to(device)
input_noise = torch.randn(1, 3, 256, 256).to(device)
input_scalar = torch.tensor([1.0]).to(device)

# Function to measure inference memory usage
def measure_inference_memory(model, *inputs):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        model(*inputs)
    torch.cuda.synchronize()
    inference_time = time.time() - start_time
    memory_usage = torch.cuda.max_memory_allocated(device)
    return memory_usage, inference_time

# Measure memory usage for UNet model
unet_memory_usage, unet_inference_time = measure_inference_memory(unet_model, input_tensor)

# Measure memory usage for Diffusion model
diffusion_memory_usage, diffusion_inference_time = measure_inference_memory(diffusion_model, input_tensor, input_noise, input_scalar)

# Print model summaries
print("UNet Model Summary:")
print(unet_summary)
print("\nDiffusion Model Summary:")
print(diffusion_summary)

# Print memory usage and inference time
print(f"\nUNet Model Memory Usage: {unet_memory_usage / (1024 ** 2):.2f} MB")
print(f"UNet Model Inference Time: {unet_inference_time:.4f} seconds")

print(f"\nDiffusion Model Memory Usage: {diffusion_memory_usage / (1024 ** 2):.2f} MB")
print(f"Diffusion Model Inference Time: {diffusion_inference_time:.4f} seconds")
