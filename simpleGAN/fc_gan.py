import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Descrimator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.desc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.05),
            nn.Linear(128, 512),
            nn.LeakyReLU(0.05),            
            nn.Linear(512, 128),
            nn.LeakyReLU(0.05),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.desc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.gen(x)

# Hyperparameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
lr = 3e-4
z_dim = 64      # 128, 256
img_dim = 28 * 28 * 1    # 768
num_epochs = 30
batch_size = 5000 # 640

disc =  Descrimator(img_dim).to(device)
gen =   Generator(z_dim, img_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        # transforms.Normalize((0.1307,), (0.3081,))
    ]
)

dataset = datasets.MNIST(root='dataset/', transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_D = optim.Adam(disc.parameters(),   lr=lr)
opt_G = optim.Adam(gen.parameters(),    lr=lr)
critertion = nn.BCELoss()

writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

step = 0
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, img_dim).to(device)
        batch_size = real.shape[0]
        
        # Train Discriminator   # max log(D(real)) + log(1-D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        
        disc_real = disc(real).view(-1)
        disc_real_loss = critertion(disc_real, torch.ones_like(disc_real))      # log(D(real))
        disc_fake = disc(fake).view(-1)
        disc_fake_loss = critertion(disc_fake, torch.zeros_like(disc_fake))     # log(1-D(G(z)))
        disc_loss = (disc_real_loss + disc_fake_loss) / 2
        
        disc.zero_grad()
        disc_loss.backward(retain_graph=True)
        opt_D.step()
        
        # Train Generator   # min log(1 - D(G(z))) -> max log(D(G(z)))
        output = disc(fake).view(-1)
        gen_loss = critertion(output,   torch.ones_like(output))
        
        gen.zero_grad()
        gen_loss.backward()
        opt_G.step()
        
        print(
            f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(loader)}] \n"
            f"Discriminator Loss: {disc_loss.item():.4f} Generator Loss: {gen_loss.item():.4f}"
        )
        # if batch_size == 0:
        #     print(
        #         f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(loader)}] \n"
        #         f"Discriminator Loss: {disc_loss.item():.4f} Generator Loss: {gen_loss.item():.4f}"
        #     )
        
        # tensorboard summary statistics
        with torch.no_grad():
            fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
            data = real.reshape(-1, 1, 28, 28)
            
            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(real, normalize=True)
            
            writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
            writer_real.add_image("Real Images", img_grid_real, global_step=step)
            
            step += 1
    
    torch.save(gen.state_dict(), f"generator-epoch{epoch+1}.pt")