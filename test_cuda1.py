import torch

device = 'cuda:9'  # Get the device from config

if torch.cuda.is_available() and device.startswith('cuda'):
    print(f"CUDA Available: {device}")
else:
    print("CPU")


print(torch.cuda.device_count())
print(torch.cuda.current_device())