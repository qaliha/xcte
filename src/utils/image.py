from PIL import Image

def load_img(filepath, resize=True):
    img = Image.open(filepath).convert('RGB')
    if resize:
        img = img.resize((256, 256), Image.BICUBIC)
    return img