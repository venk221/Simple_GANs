import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# from Project_3_kaggle import Generator     #Import Generator Class from Training
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
# # Define the device for the training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    '''
    This function is used to load the saved generator model using PyTorch.
    '''
    # model = torch.load(r'./generator.pt')
    z_dim = 100
    image_dim = 28*28
    model = Generator(z_dim, image_dim)
    model.load_state_dict(torch.load(r'./generator_weights.pt'))
    model.to(device)
    model.eval()

    return model

def generate_images(model, num_images):
    '''
    Take the model as input and generate a specified number of images.
    '''
    z_dim = 100
    images = []

    for _ in range(num_images):
        z = torch.randn(1, z_dim, device=device)
        generated_image = model(z).detach().cpu().numpy().reshape(28, 28)
        images.append(generated_image)

    return images

def plot_images(images, grid_size):
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10), tight_layout=True)

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i], cmap='gray')
        ax.axis("off")

    plt.show()
    return fig

if __name__ == "__main__":
    # Instantiate the generator model and load the saved state dictionary
    model = load_model()
    print(model)
    # Generate 25 new images
    num_images = 100
    images = generate_images(model, num_images)
    print(len(images))
    # Show the generated images in a 5x5 grid
    grid_size = 10
    fig = plot_images(images, grid_size)

    #save the plot 
    fig.savefig('./generated_images.jpg')
    plt.close(fig)