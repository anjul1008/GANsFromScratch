# Install necessary dependencies (if not already installed)
#? !pip install torch torchvision matplotlib torchmetrics scipy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import numpy as np
import matplotlib.pyplot as plt

from torch.nn.utils import spectral_norm
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 training data
batch_size = 64
dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


class Generator(nn.Module):
    def __init__(self, z_dim=128, img_channels=3, feature_maps=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: latent vector Z
            nn.Linear(z_dim, feature_maps * 8 * 4 * 4),
            nn.BatchNorm1d(feature_maps * 8 * 4 * 4),
            nn.ReLU(True),
            # Reshape to (feature_maps*8) x 4 x 4
            nn.Unflatten(1, (feature_maps * 8, 4, 4)),

            # Upsample to 8x8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # Upsample to 16x16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # Upsample to 32x32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # Final conv layer, output 3 channels
            nn.Conv2d(feature_maps, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # outputs in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_maps=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: 3 x 32 x 32
            spectral_norm(nn.Conv2d(img_channels, feature_maps, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_maps, feature_maps, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_maps * 4, feature_maps * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            spectral_norm(nn.Linear(feature_maps * 4 * 4 * 4, 1))
        )

    def forward(self, img):
        return self.net(img).view(-1)


z_dim = 128
G = Generator(z_dim=z_dim).to(device)
D = Discriminator().to(device)

lr = 2e-4
beta1, beta2 = 0.0, 0.9
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))


import torch.nn.functional as F

n_epochs = 100  # adjust as needed
print_interval = 100

for epoch in range(n_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        bs = real_imgs.size(0)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Compute discriminator output on real images
        real_validity = D(real_imgs)
        # Hinge loss for real images: max(0, 1 - D(x))
        d_loss_real = torch.mean(F.relu(1.0 - real_validity))
        
        # Generate fake images
        z = torch.randn(bs, z_dim, device=device)
        fake_imgs = G(z)
        fake_validity = D(fake_imgs.detach())
        # Hinge loss for fake images: max(0, 1 + D(G(z)))
        d_loss_fake = torch.mean(F.relu(1.0 + fake_validity))
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        # Generator loss: -D(G(z))
        gen_validity = D(fake_imgs)
        g_loss = -torch.mean(gen_validity)
        g_loss.backward()
        optimizer_G.step()

        if i % print_interval == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] Batch {i}/{len(dataloader)} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # Optionally: Save sample images during training
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            sample_z = torch.randn(64, z_dim, device=device)
            sample_imgs = G(sample_z)
            grid = make_grid(sample_imgs, normalize=True)
            plt.figure(figsize=(6, 6))
            plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
            plt.axis("off")
            plt.title(f"Epoch {epoch+1}")
            plt.show()


# Generate 10 images
G.eval()
with torch.no_grad():
    sample_z = torch.randn(10, z_dim, device=device)
    generated_imgs = G(sample_z)

# Create a grid and display
grid = make_grid(generated_imgs, nrow=5, normalize=True)
plt.figure(figsize=(8, 8))
plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
plt.axis("off")
plt.title("10 Generated Images")
plt.show()


# Initialize metrics
inception_metric = InceptionScore().to(device)
fid_metric = FrechetInceptionDistance(feature=2048).to(device)

# Compute IS for generated images
# Note: The InceptionScore metric may require multiple samples. Here we pass the generated images.
IS, _ = inception_metric(generated_imgs)
print(f"Inception Score: {IS.item():.4f}")

# For FID, we update with generated images (fake) and a batch of real images.
# Here we use one batch from the dataloader for demonstration.
real_batch, _ = next(iter(dataloader))
real_batch = real_batch.to(device)
fid_metric.update(real_batch, real=True)
fid_metric.update(generated_imgs, real=False)
FID = fid_metric.compute()
print(f"Frechet Inception Distance: {FID.item():.4f}")
