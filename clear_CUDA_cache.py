import torch

def clear_cuda_cache():
  """Clears the PyTorch CUDA cache."""
  torch.cuda.empty_cache()
  print("CUDA cache cleared.")

if __name__ == "__main__":
  clear_cuda_cache()