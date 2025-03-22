# !pip install torch torchvision numpy matplotlib scipy torch-fidelity

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from torch.autograd import grad
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 dataset
batch_size = 64
dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Class labels
class_names = dataset.classes
print(class_names)  # ['airplane', 'automobile', 'bird', 'cat', ...]

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 dataset
batch_size = 64
dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Class labels
class_names = dataset.classes
print(class_names)  # ['airplane', 'automobile', 'bird', 'cat', ...]

class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10, img_channels=3):
        super(Generator, self).__init__()
        
        self.label_embedding = nn.Embedding(num_classes, z_dim)

        self.model = nn.Sequential(
            nn.Linear(z_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, img_channels * 32 * 32),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embedding = self.label_embedding(labels)
        x = torch.cat([z, label_embedding], dim=1)
        img = self.model(x).view(-1, 3, 32, 32)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_channels=3):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, 32 * 32 * img_channels)

        self.model = nn.Sequential(
            nn.Linear(img_channels * 32 * 32 * 2, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, img, labels):
        label_embedding = self.label_embedding(labels).view(img.shape)
        x = torch.cat([img.view(img.shape[0], -1), label_embedding.view(img.shape[0], -1)], dim=1)
        return self.model(x)

def compute_gradient_penalty(D, real_images, fake_images, labels, device="cuda"):
    alpha = torch.rand(real_images.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)

    d_interpolates = D(interpolates, labels)
    fake = torch.ones(d_interpolates.size(), device=device)

    gradients = grad(outputs=d_interpolates, inputs=interpolates,
                     grad_outputs=fake, create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 100
lambda_gp = 10
n_critic = 5  # Train D more often than G
epochs = 100

# Initialize models
G = Generator(z_dim=z_dim).to(device)
D = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.9))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.9))

# Training loop
for epoch in range(epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        real_images, labels = real_images.to(device), labels.to(device)
        
        # Train Discriminator
        for _ in range(n_critic):
            z = torch.randn(batch_size, z_dim, device=device)
            fake_images = G(z, labels)
            d_real = D(real_images, labels)
            d_fake = D(fake_images.detach(), labels)

            gradient_penalty = compute_gradient_penalty(D, real_images, fake_images, labels, device)
            loss_D = -(torch.mean(d_real) - torch.mean(d_fake)) + lambda_gp * gradient_penalty
            
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        # Train Generator
        z = torch.randn(batch_size, z_dim, device=device)
        fake_images = G(z, labels)
        loss_G = -torch.mean(D(fake_images, labels))

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")


z = torch.randn(10, z_dim, device=device)
labels = torch.full((10,), 1, dtype=torch.long, device=device)  # Class 1 = automobile
fake_images = G(z, labels)

# Convert to numpy and denormalize
grid = make_grid(fake_images.cpu().detach(), normalize=True)
plt.figure(figsize=(10, 10))
plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
plt.axis("off")
plt.show()


# Compute Inception Score (IS)
is_score = InceptionScore().to(device)
IS = is_score(fake_images)[0].item()
print(f"Inception Score: {IS:.4f}")

# Compute Frechet Inception Distance (FID)
fid = FrechetInceptionDistance(feature=2048).to(device)
fid.update(fake_images, real=False)
fid.update(real_images, real=True)
FID = fid.compute().item()
print(f"Frechet Inception Distance: {FID:.4f}")
