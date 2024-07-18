import wget
import os
import scipy.io
import numpy as np
import pandas as pd
import base64
import torch
from torchvision.transforms import ToTensor, Normalize
import sys
from tqdm import tqdm  # Import tqdm for progress bar

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from diffusion_denoising.diffusion_RDUnet import RDUNet_T, DiffusionModel  # Ensure your diffusion model script is correctly imported

# Model parameters
base_filters = 32
timesteps = 20

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
model = DiffusionModel(RDUNet_T(base_filters=base_filters), timesteps=timesteps).to(device)
checkpoint_path = 'checkpoints/diffusion_RDUnet_model_checkpointed_epoch_43.pth'  # Adjust path as needed
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])  # Only load the model parameters
model.eval()

# Normalization transform
normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def my_srgb_denoiser(x):
    """Denoise the sRGB image using the trained diffusion model."""
    # Convert the numpy array to a PyTorch tensor and normalize
    x = ToTensor()(x).unsqueeze(0).to(device)
    x = normalize(x)

    with torch.no_grad():
        denoised = model.improved_sampling(x)

    # Denormalize and convert back to numpy array
    denoised = denoised.squeeze().cpu().numpy().transpose(1, 2, 0)
    denoised = (denoised + 1) / 2  # Convert from [-1, 1] to [0, 1]
    denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
    
    return denoised

def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

def base64string_to_array(base64string, array_dtype, array_shape):
    decoded_bytes = base64.b64decode(base64string)
    decoded_array = np.frombuffer(decoded_bytes, dtype=array_dtype)
    decoded_array = decoded_array.reshape(array_shape)
    return decoded_array

# Download input file, if needed.
url = 'https://competitions.codalab.org/my/datasets/download/0d8a1e68-155d-4301-a8cd-9b829030d719'
input_file = 'evaluate_SIDD/BenchmarkNoisyBlocksSrgb.mat'
if os.path.exists(input_file):
    print(f'{input_file} exists. No need to download it.')
else:
    print('Downloading input file BenchmarkNoisyBlocksSrgb.mat...')
    wget.download(url, input_file)
    print('Downloaded successfully.')

# Read inputs.
key = 'BenchmarkNoisyBlocksSrgb'
inputs = scipy.io.loadmat(input_file)
inputs = inputs[key]
print(f'inputs.shape = {inputs.shape}')
print(f'inputs.dtype = {inputs.dtype}')

# Denoising.
output_blocks_base64string = []
total_blocks = inputs.shape[0] * inputs.shape[1]
progress_bar = tqdm(total=total_blocks, desc='Denoising')

for i in range(inputs.shape[0]):
    for j in range(inputs.shape[1]):
        in_block = inputs[i, j, :, :, :]
        out_block = my_srgb_denoiser(in_block)
        assert in_block.shape == out_block.shape
        assert in_block.dtype == out_block.dtype
        out_block_base64string = array_to_base64string(out_block)
        output_blocks_base64string.append(out_block_base64string)
        progress_bar.update(1)  # Update the progress bar

progress_bar.close()  # Close the progress bar

# Save outputs to .csv file.
output_file = 'SubmitSrgb.csv'
print(f'Saving outputs to {output_file}')
output_df = pd.DataFrame()
n_blocks = len(output_blocks_base64string)
print(f'Number of blocks = {n_blocks}')
output_df['ID'] = np.arange(n_blocks)
output_df['BLOCK'] = output_blocks_base64string

output_df.to_csv(output_file, index=False)

# TODO: Submit the output file SubmitSrgb.csv at 
# kaggle.com/competitions/sidd-benchmark-srgb-psnr
print('TODO: Submit the output file SubmitSrgb.csv at')
print('kaggle.com/competitions/sidd-benchmark-srgb-psnr')

print('Done.')
