import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
    
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
device = torch.device("cpu")
print(device)
z_dim = 64      # 128, 256
img_dim = 28 * 28 * 1    # 768
batch_size = 1 # 640

chkp = 'generator-epoch30.pt'
gen =   Generator(z_dim, img_dim).to(device)
gen.load_state_dict(torch.load(chkp, map_location=device))


fixed_noise = torch.randn((batch_size, z_dim)).to(device)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        # transforms.Normalize((0.1307,), (0.3081,))
    ]
)

noise = torch.randn(batch_size, z_dim).to(device)
fake = gen(noise)
fake = fake.reshape(28, 28).detach().numpy()
print(fake.shape)
# plt.imshow(noise)
plt.imshow(fake, cmap='Greys')
plt.show()