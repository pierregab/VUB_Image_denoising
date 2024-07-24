import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist

def sample_uniform(num_samples, timesteps):
    return torch.randint(0, timesteps + 1, (num_samples,)).float()

def sample_biased(num_samples, timesteps, alpha=1.5):
    beta_dist = dist.Beta(alpha, 1.0)
    return beta_dist.sample((num_samples,)) * timesteps

def plot_distributions(num_samples=10000, timesteps=20):
    # Generate samples
    uniform_samples = sample_uniform(num_samples, timesteps)
    biased_samples = sample_biased(num_samples, timesteps)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot uniform distribution
    ax1.hist(uniform_samples.numpy(), bins=timesteps+1, range=(0, timesteps), density=True)
    ax1.set_title('Uniform Distribution')
    ax1.set_xlabel('Timestep (t)')
    ax1.set_ylabel('Density')

    # Plot biased distribution
    ax2.hist(biased_samples.numpy(), bins=50, density=True)
    ax2.set_title('Biased Distribution')
    ax2.set_xlabel('Timestep (t)')
    ax2.set_ylabel('Density')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_distributions()