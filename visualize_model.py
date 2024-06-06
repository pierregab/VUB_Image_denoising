import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchsummary
from torchviz import make_dot
from VUB_Image_denoising.RCA_GAN.paper_gan import Generator, Discriminator

def print_model_summary(model, input_size):
    print(f"\nSummary of {model.__class__.__name__}:")
    torchsummary.summary(model, input_size=input_size)

# Assuming your input size is (1, 256, 256) for grayscale images
input_size = (1, 256, 256)

# Instantiate your models
in_channels = 1
out_channels = 1
generator = Generator(in_channels, out_channels)
discriminator = Discriminator(in_channels)

# Print summaries
print_model_summary(generator, input_size)
print_model_summary(discriminator, input_size)

# Visualize the generator model
dummy_input = torch.randn(1, 1, 256, 256)  # Batch size of 1, 1 channel, 256x256 image
output = generator(dummy_input)

# Create a graph of the model
dot = make_dot(output, params=dict(generator.named_parameters()))

# Save the graph to a file
dot.format = 'png'
dot.render('generator_model')

