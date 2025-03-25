# Install required packages if not already installed:
#? !pip install torch torchvision matplotlib torchmetrics scipy

# REF:https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# For evaluation metrics
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

####################################
# 1. Data Loading and Preprocessing
####################################

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
])

batch_size = 64
dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

####################################
# 2. Self-Attention Module
####################################

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W)            # B x C' x N
        proj_key   = self.key_conv(x).view(B, -1, H * W)                # B x C' x N
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)         # B x N x N
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, H * W)              # B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

####################################
# 3. Generator Architecture
####################################
# This generator first projects a latent vector into a 4x4 feature map and then upsamples
# to 8x8, 16x16 (inserting a self-attention block) and finally to 32x32.

class Generator(nn.Module):
    def __init__(self, z_dim=128, feature_maps=64, img_channels=3):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        self.proj = nn.Sequential(
            nn.Linear(z_dim, feature_maps * 8 * 4 * 4),
            nn.BatchNorm1d(feature_maps * 8 * 4 * 4),
            nn.ReLU(True)
        )
        
        self.net = nn.Sequential(
            # Input: (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # Upsample to 16x16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # Insert Self-Attention at 16x16 resolution
            SelfAttention(feature_maps * 2),
            
            # Upsample to 32x32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1),     # 32x32
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # Final convolution to get image channels
            nn.Conv2d(feature_maps, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, z):
        x = self.proj(z)
        x = x.view(z.size(0), -1, 4, 4)
        img = self.net(x)
        return img

####################################
# 4. Discriminator Architecture
####################################
# The discriminator downsamples the image through several convolutional layers with spectral normalization.
# A self-attention block is inserted at the 16x16 resolution.

from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, feature_maps=64, img_channels=3):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: 3 x 32 x 32
            spectral_norm(nn.Conv2d(img_channels, feature_maps, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsample to 16x16
            spectral_norm(nn.Conv2d(feature_maps, feature_maps, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Insert Self-Attention at 16x16
            SelfAttention(feature_maps),
            
            # Downsample to 8x8
            spectral_norm(nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsample to 4x4
            spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(feature_maps * 4, feature_maps * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            spectral_norm(nn.Linear(feature_maps * 4 * 4 * 4, 1))
        )
        
    def forward(self, img):
        validity = self.net(img)
        return validity.view(-1)

####################################
# 5. Initialize Models and Optimizers
####################################

z_dim = 128
G = Generator(z_dim=z_dim).to(device)
D = Discriminator().to(device)

# Use TTUR or same lr for both (here we use same lr for simplicity)
lr = 2e-4
beta1, beta2 = 0.0, 0.9
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

####################################
# 6. Training Loop (Hinge Loss)
####################################

import time
n_epochs = 50  # Adjust number of epochs as needed
print_interval = 100

for epoch in range(n_epochs):
    start_time = time.time()
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        bs = real_imgs.size(0)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Hinge loss for real images: max(0, 1 - D(x))
        real_validity = D(real_imgs)
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
        gen_validity = D(fake_imgs)
        g_loss = -torch.mean(gen_validity)
        g_loss.backward()
        optimizer_G.step()
        
        if i % print_interval == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] Batch {i}/{len(dataloader)} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
    
    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1} completed in {elapsed:.2f} sec.")

####################################
# 7. Generate and Display 10 New Images
####################################

G.eval()
with torch.no_grad():
    sample_z = torch.randn(10, z_dim, device=device)
    gen_imgs = G(sample_z).cpu()

grid = make_grid(gen_imgs, nrow=5, normalize=True)
plt.figure(figsize=(8, 8))
plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
plt.axis("off")
plt.title("10 Generated Images")
plt.show()

####################################
# 8. Evaluation: Inception Score (IS) and FID
####################################

# Note: For robust metrics one should use many generated images.
# Here we compute a rough estimate using a larger set.
inception_metric = InceptionScore().to(device)
fid_metric = Frechet
