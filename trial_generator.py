import argparse
import torch.optim as optim
import torch
from torchsummary import summary
from src.networks.discriminator import Discriminator
from src.networks.generator import Generator, trial
from src.model import Model
from src.utils.compression import _compress
# from loader import normalize

generator = Generator()

summary(generator, (3, 256, 256), 1)
# torch.save(generator, 'generator.pt')

trial()
exit()


parser = argparse.ArgumentParser(description='Compressing')
parser.add_argument('--cuda', action='store_true', help='use cuda?')

opt = parser.parse_args()

model = Model(3, opt)

D = Discriminator()
G = Generator()

squared_difference = torch.nn.MSELoss(reduction='mean')

opt_generator = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_discriminator = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Parameters encoder before
before = list(G.parameters())[0].clone()
before_d = list(D.parameters())[0].clone()

image = torch.randn((1, 3, 256, 256))
compressed_image = torch.randn((1, 3, 256, 256))

model.set_requires_grad(G, False)

# expanded = G(compressed_image)

# model.set_requires_grad(D, True)
# opt_discriminator.zero_grad()

# # Update discriminator
# fake_ab = D(torch.cat((compressed_image, expanded), 1).detach())
# loss_d_fake = model.gan_loss(fake_ab, False)

# real_ab = D(torch.cat((compressed_image, image), 1))
# loss_d_real = model.gan_loss(real_ab, True)

# discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

# discriminator_loss.backward()
# opt_discriminator.step()

# model.set_requires_grad(D, False)
# opt_generator.zero_grad()

# # Update generator
# fake_ab = D(torch.cat((compressed_image, expanded), 1))

# gan_losses = model.gan_loss(fake_ab, True)
# decoder_losses = model.squared_difference(expanded, image) * 0.5
# # perceptual_losses = model.perceptual_loss(expanded, image)

# # assert(expanded.requires_grad)
# # assert(image.requires_grad)

# # save_img(expanded.detach().squeeze(0).cpu(), 'interm/generated.png')
# # save_img(image.detach().squeeze(0).cpu(), 'interm/inputed.png')
# # save_img(compressed_image.detach().squeeze(0).cpu(), 'interm/compress.png')

# generator_losses = gan_losses + decoder_losses

# # discriminator_loss, generator_losses = model.gd_training(compressed_image, image)

# # Backward losses and step optimizer
# # Zero gradient

# generator_losses.backward()
# opt_generator.step()

expanded = model.Generator(compressed_image)

model.set_requires_grad(model.Discriminator, True)

opt_discriminator.zero_grad()

D_in = torch.cat((image, expanded.detach()), dim=1)

D_out, D_out_logits = model.Discriminator(D_in)
D_out = torch.squeeze(D_out)
D_out_logits = torch.squeeze(D_out_logits)

D_real, D_gen = torch.chunk(D_out, 2, dim=0)
D_real_logits, D_gen_logits = torch.chunk(D_out_logits, 2, dim=0)

discriminator_loss = model.gan_loss_hf(
    D_real, D_gen, D_real_logits, D_gen_logits, 'discriminator_loss')

# # Update discriminator
# fake_ab = model.Discriminator(
#     torch.cat((compressed_image, expanded), 1).detach())
# loss_d_fake = model.gan_loss(fake_ab, False)

# real_ab = model.Discriminator(
#     torch.cat((compressed_image, image), 1))
# loss_d_real = model.gan_loss(real_ab, True)

# discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

discriminator_loss.backward()
opt_discriminator.step()

opt_generator.zero_grad()

model.set_requires_grad(model.Discriminator, False)

D_in = torch.cat((image, expanded), dim=1)

D_out, D_out_logits = model.Discriminator(D_in)
D_out = torch.squeeze(D_out)
D_out_logits = torch.squeeze(D_out_logits)

D_real, D_gen = torch.chunk(D_out, 2, dim=0)
D_real_logits, D_gen_logits = torch.chunk(D_out_logits, 2, dim=0)

gan_losses = model.gan_loss_hf(
    D_real, D_gen, D_real_logits, D_gen_logits, 'generator_loss')

decoder_losses = model.restruction_loss(expanded, image)
perceptual_losses = model.perceptual_loss(
    expanded, image, normalize=False)
generator_losses = .15 * gan_losses + \
    (decoder_losses + perceptual_losses)

generator_losses.backward()
opt_generator.step()

# Parameters encoder after
after = list(G.parameters())[0].clone()
after_d = list(D.parameters())[0].clone()

assert(torch.all(torch.eq(before_d, after_d)) == False)
assert(torch.all(torch.eq(before, after)) == False)
print('Different?')
