import torch
import matplotlib.pyplot as plt
import numpy as np

def exponential_t_sampling(batch_size, timesteps, lambda_param=1.0):
    # Generate exponentially distributed samples between 0 and 1
    t = torch.distributions.Exponential(lambda_param).sample((batch_size,))
    
    # Normalize to [0, 1] range
    t = 1 - torch.exp(-t)
    
    # Scale to timesteps
    t = t * timesteps
    return t.round().long()

def visualize_bias_distribution(timesteps, lambda_params, num_samples=1000000):
    plt.figure(figsize=(15, 8))
    
    for lambda_param in lambda_params:
        samples = exponential_t_sampling(num_samples, timesteps, lambda_param).numpy()
        counts, bin_edges = np.histogram(samples, bins=timesteps, range=(0, timesteps), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.plot(bin_centers, counts, label=f'Exponential (λ={lambda_param:.2f})')
    
    plt.axvline(timesteps, color='red', linestyle='dashed', linewidth=1, label='Noisiest Timestep')
    plt.xlabel('Timesteps')
    plt.ylabel('Probability Density')
    plt.title('Probability Distribution of t using Exponential Distribution')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    timesteps = 70
    lambda_params = [0.5]  # Different lambda values for comparison
    visualize_bias_distribution(timesteps, lambda_params)