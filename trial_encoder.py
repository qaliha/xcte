import torch.optim as optim
import torch

from src.networks.encoder import Encoder
from src.networks.generator import Generator
from src.utils.compression import _compress
from loader import normalize

E = Encoder()
G = Generator()

squared_difference = torch.nn.MSELoss(reduction='mean')

opt_encoder = optim.Adam(E.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Parameters encoder before
before = list(E.parameters())[0].clone()

for name, param in E.named_parameters():
    print(name)

input = torch.randn((2, 3, 256, 256), requires_grad=True)

image = E(input)
# image = _compress(image, 3)

# Normalize the output first
image = normalize(image)

for param in G.parameters():
    param.requires_grad = False

image = G(image)

print(input.requires_grad)
print(image.requires_grad)

compression_losses = squared_difference(image, input) * 0.5

opt_encoder.zero_grad()
compression_losses.backward()
opt_encoder.step()

# Parameters encoder after
after = list(E.parameters())[0].clone()

assert(torch.all(torch.eq(before, after)) == False)
print('Different?')
