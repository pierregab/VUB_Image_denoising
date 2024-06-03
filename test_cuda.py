import torch

def check_cuda_with_torch():
    if torch.cuda.is_available():
        print("CUDA is available with PyTorch.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available with PyTorch.")

if __name__ == "__main__":
    check_cuda_with_torch()
