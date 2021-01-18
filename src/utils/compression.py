import torch

def compress(tensor, bit):
    max_val = 2**bit - 1
    tensor = torch.clamp(tensor, 0.0, 1.0) * max_val
    tensor = torch.round(tensor)
    tensor = tensor / max_val
    return tensor