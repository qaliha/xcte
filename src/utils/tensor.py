import numpy as np
from PIL import Image

def tensor2img(tensor):
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    tensor = np.squeeze(tensor)
    tensor = np.moveaxis(tensor, 0, 2)
    tensor = tensor * 255
    tensor = tensor.clip(0, 255).astype(np.uint8)
    
    img = Image.fromarray(tensor)
    return img