import torch.optim as optim
import torch

from src.networks.encoder import Encoder
from src.networks.generator import Generator
from src.utils.compression import compress
from loader import normalize

Encoder = Encoder()
Generator = Generator()

squared_difference = torch.nn.MSELoss(reduction='mean')

opt_encoder = optim.Adam(list(Encoder.parameters()) + list([Encoder.connection_weights]), lr=0.0002, betas=(0.5, 0.999))

before = list(Encoder.parameters())[0].clone()

image = torch.randn((2, 3, 256, 256))

compressed = Encoder(image)
compressed = compress(compressed, 3)

# Normalize the output first
normalized = normalize(compressed)

for param in Generator.parameters():
    param.requires_grad = False

expanded = Generator(normalized)

compression_losses = squared_difference(expanded, image) * 0.5

opt_encoder.zero_grad()
compression_losses.backward()
opt_encoder.step()

after = list(Encoder.parameters())[0].clone()

assert(torch.all(torch.eq(before, after)) == False)
print('Different?')
