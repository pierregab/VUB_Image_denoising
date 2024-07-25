import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random
import sys
from PIL import Image
import cv2
from unet.unet import UNetModel  # Ensure correct import path
import utils.utils_image as util
import utils.utils_sampling as utils_sampling
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Set the appropriate device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

os.environ["CUDA_VISIBLE_DEVICES"] = '2' if device == 'cuda' else ''

def data_transform(X):
    return 2 * X - 1.0

def data_transform_reverse(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class Train(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device_id = config['device_id']
        self.device = torch.device(self.device_id)

        self.model = UNetModel()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.config['save']['ddpm_checkpoint'], map_location=self.device))

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        betas = get_beta_schedule(
            beta_schedule=config['diffusion']['beta_schedule'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'],
            num_diffusion_timesteps=config['diffusion']['num_diffusion_timesteps'], )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        print('____________________________prepare_over____________________________')

    def sample_image(self,
                     x,
                     last=True,
                     eta=0.8,
                     difussion_times=500
                     ):
        #______________________embedding for gaussian___________________
        e = torch.randn_like(x[:, :, :, :])
        t = torch.ones(x.shape[0]).to(self.device)
        t = t*(difussion_times-1)
        t = torch.tensor(t,dtype=torch.int)
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x_N = x[:, :, :, :] * a.sqrt()
        #____________________________  end  ____________________________
        
        skip = difussion_times // self.config['diffusion']['sampling_timesteps']
        seq = range(0, difussion_times, skip)
        seq=list(seq[:-1])+[difussion_times-1]  #a liilte different from ddim

        xs = utils_sampling.generalized_steps(x=x_N,
                                            seq=seq,
                                            model=self.model,
                                            b=self.betas,
                                            eta=eta)

        if last:
            xs = xs[0][-1]
        return xs

    def MMSE(self,x,y,sigma):
        tmp=-((y-x)**2)/2.0/sigma/sigma 
        tmp= torch.exp(tmp)
        likelihoods= tmp/ np.sqrt((2.0*np.pi*sigma))
        mseEst = torch.sum(likelihoods*x,dim=0,keepdim=True)[0,...] # Sum up over all samples
        mseEst/= torch.sum(likelihoods,dim=0,keepdim=True)[0,...] # Normalize
        
        return mseEst
        
    def eval_ddpm(self,
                  image_path,
                  eta=0.8,
                  eval_time=1,
                  difussion_times=500,
                  mmse_average=False,
                  sigma=50
                  ):
        idx = 0
        with torch.no_grad():
            # Load image
            img = Image.open(image_path).convert('RGB')
            img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
            c, h, w = img.shape[1:]

            # Add Gaussian noise to the image
            noise = img + torch.randn_like(img) * (sigma / 255.0)
            noise = torch.clamp(noise, 0.0, 1.0)
            
            # Divide the image into patches
            patch_size = 256
            step = patch_size // 2  # Overlapping patches
            denoise = torch.zeros_like(noise)
            weight_map = torch.zeros_like(noise)
            
            for i in tqdm(range(0, h-patch_size+1, step), desc="Processing rows"):
                for j in tqdm(range(0, w-patch_size+1, step), desc="Processing columns", leave=False):
                    noise_patch = noise[:, :, i:i+patch_size, j:j+patch_size]
                    
                    ######occupy gpu
                    if mmse_average:
                        self.MMSE(noise_patch, torch.unsqueeze(noise_patch, dim=0).repeat(eval_time,1,1,1,1), sigma/255)
                    ######occupy gpu
                    
                    denoise_patch = torch.zeros_like(noise_patch).to('cpu')
                    
                    for _ in range(eval_time):
                        x = torch.randn_like(noise_patch)
                        denoise_patch += self.sample_image(x=data_transform(noise_patch),
                                                           eta=eta,
                                                           difussion_times=difussion_times,
                                                           )
                    denoise_patch = data_transform_reverse(denoise_patch / eval_time).to(self.device)

                    if mmse_average:
                        denoise_patch = self.MMSE(denoise_patch, torch.unsqueeze(noise_patch, dim=0).repeat(eval_time,1,1,1,1), sigma/255)
                    else:
                        denoise_patch = torch.mean(denoise_patch, dim=0)
                    
                    denoise[:, :, i:i+patch_size, j:j+patch_size] += denoise_patch
                    weight_map[:, :, i:i+patch_size, j:j+patch_size] += 1
            
            denoise /= weight_map

            noise = noise[:, :, :h, :w]
            denoise = denoise[:, :, :h, :w]

            denoise_img = denoise[0, :, :, :].permute(1, 2, 0).cpu().numpy()
            noise_img = noise[0, :, :, :].permute(1, 2, 0).cpu().numpy()
            gt_img = img[0, :, :, :].permute(1, 2, 0).cpu().numpy()

            # Save the results
            denoise_img_path = self.config['save']['photo_path'] + 'denoise_' + os.path.basename(image_path)
            noise_img_path = self.config['save']['photo_path'] + 'noise_' + os.path.basename(image_path)
            gt_img_path = self.config['save']['photo_path'] + 'gt_' + os.path.basename(image_path)

            cv2.imwrite(denoise_img_path, (denoise_img * 255).astype(np.uint8))
            cv2.imwrite(noise_img_path, (noise_img * 255).astype(np.uint8))
            cv2.imwrite(gt_img_path, (gt_img * 255).astype(np.uint8))

            # Plot and save the images
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(noise_img)
            axs[0].set_title('Noisy Image')
            axs[0].axis('off')
            
            axs[1].imshow(gt_img)
            axs[1].set_title('Ground Truth')
            axs[1].axis('off')
            
            axs[2].imshow(denoise_img)
            axs[2].set_title('Denoised Image')
            axs[2].axis('off')
            
            plt.savefig(self.config['save']['photo_path'] + 'comparison_' + os.path.basename(image_path))
            plt.show()

        print('Finish!')

# You can also change your options here
option = {
    'data': {
        'test_n_channels': 3,
        'test_H_size': 256,                                 # fixed, image size for ImageNet
        'test_batch_size': 1,
        'test_num_workers': 0,
        'test_opt': 'ImageNet',                             # Dataset
        'noise_sigma': 150                                  # test noise level for Gaussian noise
    },
    'model': {
        'type': "openai",                                   # fixed
    },
    'diffusion': {
        'beta_schedule': 'linear',                          # fixed
        'beta_start': 0.0001,                               # fixed
        'beta_end': 0.02,                                   # fixed
        'num_diffusion_timesteps': 1000,                    # fixed
        'sampling_timesteps': 1                             # fixed
    },
    'save': {
        'photo_path': './results/photo_temp/',
        'ddpm_checkpoint': './pre-trained/256x256_diffusion_uncond.pt'  # the path of the pre-trained diffusion model
    },
    'device_id': device,
}

import argparse
parser = argparse.ArgumentParser(description='Gaussian Color Denoising using DMID')

parser.add_argument('--image_path', default='../dataset/DIV2K_train_HR.nosync/0001.png', type=str, help='Path to the input image')
parser.add_argument('--test_sigma', default=250, type=int, help='50/100/150/200/250/...')
parser.add_argument('--S_t', default=10, type=int, help='Sampling times in one inference')
parser.add_argument('--R_t', default=1, type=int, help='Repetition times of multiple inferences')
parser.add_argument('--mmse_average', default=False, action='store_true', help='MMSE average for better perceptual quality')
args = parser.parse_args()

N_list = [33, 57, 115, 215, 291, 348, 393]
noise_sigma_list = [15, 25, 50, 100, 150, 200, 250]

if args.S_t > 3:
    seed = 2024
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print('Random seed:', seed)

option['data']['noise_sigma'] = args.test_sigma
option['diffusion']['sampling_timesteps'] = args.S_t
N = N_list[noise_sigma_list.index(args.test_sigma)]
os.makedirs('./results/photo_temp', exist_ok=True)
option['save']['photo_path'] = './results/photo_temp/'

TRAIN = Train(config=option)
print(option)

TRAIN.eval_ddpm(
    image_path=args.image_path,
    eval_time=args.R_t,
    difussion_times=N,
    sigma=args.test_sigma,
    mmse_average=args.mmse_average,
)
