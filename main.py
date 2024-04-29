import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Hyperparameters
batch_size = 128
lr = 0.0002
epochs = 200
z_dim = 100
image_size = 28 * 28
d_losses = []
g_losses = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self, z_dim, image_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

netG = Generator(z_dim, image_size).to(device)

class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


netD = Discriminator(image_size).to(device)


optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        current_batch_size = images.size(0)
        real_images = images.view(current_batch_size, -1).to(device)
        real_labels = torch.ones(current_batch_size, 1).to(device)
        optimizerD.zero_grad()
        outputs = netD(real_images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()
        z = torch.randn(current_batch_size, z_dim).to(device)
        fake_images = netG(z)
        fake_labels = torch.zeros(current_batch_size, 1).to(device)
        outputs = netD(fake_images.detach())
        d_loss = criterion(outputs, fake_labels)
        d_loss.backward()
        optimizerD.step()
        optimizerG.zero_grad()
        outputs = netD(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizerG.step()
        d_losses.append(d_loss_real.item() + d_loss.item())
        g_losses.append(g_loss.item())

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], D Loss: {d_loss_real.item() + d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

torch.save(netG, './generator.pt')
torch.save(netG.state_dict(), 'generator_weights.pt')
torch.save(netD.state_dict(), 'discriminator.pt')

plt.figure(figsize=(10,5))
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Discriminator Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Generator Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Losses')
plt.legend()
plt.show()

np.random.seed(504)
h = w = 28
num_gen = 100
z = np.random.normal(size=[num_gen, z_dim])
z = torch.from_numpy(z).float().to(device)
generated_images = netG(z)
n = int(np.sqrt(num_gen))
fig, axes = plt.subplots(n, n, figsize=(8, 8))
for i in range(n):
    for j in range(n):
        axes[i, j].imshow(generated_images[i * n + j].cpu().detach().numpy().reshape(28, 28), cmap='gray')
        axes[i, j].axis('off')
plt.show()