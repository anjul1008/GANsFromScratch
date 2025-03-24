# Install required packages if not already installed:
# !pip install torch torchvision matplotlib torchmetrics scipy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# For evaluation metrics
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

r'''
Explanation
Data Preparation:
    - CIFAR‑10 images are loaded and normalized to the range [–1, 1].

Self‑Attention Module:
    - A simple self‑attention block is defined. This block is inserted into both the generator and discriminator (after some down-/up‑sampling layers) so that the network can capture long‑range dependencies.

Generator:
    - The generator first projects a latent vector (of dimension 128) into a small feature map (4×4) and then upsamples to 32×32. A self‑attention layer is inserted at the 8×8 stage.

Discriminator:
    - The discriminator uses convolutional blocks to downsample the image from 32×32 to a 4×4 feature map and includes a self‑attention layer after the first downsampling stage.

TTUR:
    - Two different learning rates are used: 1e‑4 for the generator and 4e‑4 for the discriminator.

Training:
    - The network is trained using hinge loss. The discriminator is updated with a loss that penalizes real images with low scores and fake images with high scores; the generator is updated to fool the discriminator.

Generation and Evaluation:
    - After training, 10 images are generated and displayed. Then, the Inception Score and FID are computed using a subset of generated images and one batch of real images for a rough evaluation.

'''


####################################
# 1. Data Loading and Preprocessing
####################################

# CIFAR-10 images are scaled to [-1, 1] since our generator will use tanh as final activation.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        # Query, key, and value transformations
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, W, H = x.size()
        # Shape: B x (C//8) x (W*H)
        proj_query = self.query_conv(x).view(B, -1, W * H)
        # Shape: B x (C//8) x (W*H)
        proj_key   = self.key_conv(x).view(B, -1, W * H)
        # Compute attention map
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # B x (W*H) x (W*H)
        attention = torch.softmax(energy, dim=-1)
        # Shape: B x C x (W*H)
        proj_value = self.value_conv(x).view(B, -1, W * H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        out = self.gamma * out + x
        return out

####################################
# 3. Generator with Self-Attention
####################################

class Generator(nn.Module):
    def __init__(self, z_dim=128, feature_maps=64, img_channels=3):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        # Project and reshape
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
            
            # Self-Attention on 8x8 features (optional, can also be applied at 16x16)
            SelfAttention(feature_maps * 4),
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # You may add a second self-attention block here if desired:
            # SelfAttention(feature_maps * 2),
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            nn.Conv2d(feature_maps, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, z):
        x = self.proj(z)
        x = x.view(z.size(0), -1, 4, 4)
        img = self.net(x)
        return img

####################################
# 4. Discriminator with Self-Attention
####################################

class Discriminator(nn.Module):
    def __init__(self, feature_maps=64, img_channels=3):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(img_channels, feature_maps, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps, feature_maps, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.2, inplace=True),
            
            # Self-Attention on 16x16 features
            SelfAttention(feature_maps),
            
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 2, feature_maps * 2, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 4, feature_maps * 4, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            nn.Linear(feature_maps * 4 * 4 * 4, 1)
        )
        
    def forward(self, img):
        validity = self.net(img)
        return validity.view(-1)

####################################
# 5. Initialize Models and Optimizers (TTUR)
####################################

z_dim = 128
G = Generator(z_dim=z_dim).to(device)
D = Discriminator().to(device)

# TTUR: Different learning rates for generator and discriminator
lr_G = 1e-4
lr_D = 4e-4
beta1, beta2 = 0.0, 0.9

optimizer_G = optim.Adam(G.parameters(), lr=lr_G, betas=(beta1, beta2))
optimizer_D = optim.Adam(D.parameters(), lr=lr_D, betas=(beta1, beta2))

####################################
# 6. Training Loop (Using Hinge Loss)
####################################

n_epochs = 50   # adjust epochs as needed
print_interval = 100

for epoch in range(n_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        bs = real_imgs.size(0)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Hinge loss on real images: max(0, 1 - D(x))
        real_validity = D(real_imgs)
        d_loss_real = torch.mean(F.relu(1.0 - real_validity))
        
        # Generate fake images
        z = torch.randn(bs, z_dim, device=device)
        fake_imgs = G(z)
        fake_validity = D(fake_imgs.detach())
        # Hinge loss on fake images: max(0, 1 + D(G(z)))
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
# 8. Evaluation: IS and FID Scores
####################################

# Create metrics objects (using torchmetrics)
# Note: In practice, these metrics are computed over many samples.
inception_metric = InceptionScore().to(device)
fid_metric = FrechetInceptionDistance(feature=2048).to(device)

# For demonstration, we update the metrics with a batch of generated images.
# In a proper evaluation, you would use many samples.

# Generate many images for more robust evaluation
num_eval = 500  # number of images for evaluation
all_gen_imgs = []
for _ in range(num_eval // batch_size):
    z = torch.randn(batch_size, z_dim, device=device)
    with torch.no_grad():
        imgs = G(z)
    all_gen_imgs.append(imgs)
all_gen_imgs = torch.cat(all_gen_imgs, dim=0)

# Compute Inception Score (IS)
IS, _ = inception_metric(all_gen_imgs)
print(f"Inception Score: {IS.item():.4f}")

# For FID, we need a batch of real images as well.
real_batch, _ = next(iter(dataloader))
real_batch = real_batch.to(device)

fid_metric.update(real_batch, real=True)
fid_metric.update(all_gen_imgs[:real_batch.size(0)], real=False)
FID = fid_metric.compute()
print(f"Frechet Inception Distance: {FID.item():.4f}")
