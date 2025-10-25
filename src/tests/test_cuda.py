import torch, bitsandbytes as bnb

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current CUDA device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))
print("bitsandbytes backend:", bnb.__version__)
