import pandas as pd
import numpy as np
import os
import random
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import datetime

# Set the seed for rep
seed = 1618
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
latent_dim = 64 # dimensionality of noise vector
batch_size = 16
lr = 2e-4
num_epochs = 50

# Import the data
df = pd.read_csv("Datasets/ieee-fraud-detection/test_transaction.csv", nrows=1000)

# Scale the features for GAN
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Create a class for easy PyTorch implementation
class IrisDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
# Create dataset and loader
iris_dataset = IrisDataset(scaled_data)
data_loader = DataLoader(iris_dataset, batch_size=batch_size, shuffle=False)

# GAN architecture
# Start by the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim): # output_dim: one for each feature we want to generate
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        return self.model(z)

# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim): # input_dim: number of features in the input
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid() # output a probability between 0 and 1
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize generator and dicriminator
generator = Generator(latent_dim=latent_dim, output_dim=data.shape[1]).to(device)
discriminator = Discriminator(input_dim=data.shape[1]).to(device)

# Training loop
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(num_epochs):
    for i, real_data in enumerate(data_loader):
        real_data = real_data.to(device)
        # Train the discriminator first
        optimizer_D.zero_grad()
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # forward pass on real data
        d_real = discriminator(real_data)
        d_real_loss = criterion(d_real, real_labels)

        # train d on fake data
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_data = generator(z)

        # forward pass on fake data
        d_fake = discriminator(fake_data.detach()) # detach to avoid gradient flow back to G
        d_fake_loss = criterion(d_fake, fake_labels)
         
        # Total loss
        d_loss = d_real_loss + d_fake_loss
        
        # backprop d
        d_loss.backward()
        optimizer_D.step()

        # Train the Generator
        optimizer_G.zero_grad()

        # Generate fake data
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_data = generator(z)

        # get dicriminator output on these fakes but now we want them all to be labbeled as real
        d_fake_for_G = discriminator(fake_data)

        # generator loss to understand how well g fool d, we want d(fake) to be one
        g_loss = criterion(d_fake_for_G, real_labels)

        # backprop g
        g_loss.backward()
        optimizer_G.step()

        if i % 50 == 0:
            print(f"Epoch {epoch}/{num_epochs}"
            f" | Batch {i}/{len(data_loader)}"
            f" | Discriminator loss: {d_loss.item():.4f}"
            f" | Generator loss: {g_loss.item():.4f}")
    

# generate a small sample of data to see generator progress
test_noise = torch.randn(5, latent_dim, device=device)
synthetic_samples = generator(test_noise).detach().cpu().numpy()
# reverse transformation
synthetic_samples_original_scales = scaler.inverse_transform(synthetic_samples)
print(synthetic_samples_original_scales)
print(data)

# Compare the data
real_mean = np.mean(data, axis=0)
real_std = np.std(data, axis=0)

# Synthetic data
z_test = torch.randn(150, latent_dim, device=device)
fake_large = generator(z_test).detach().cpu().numpy()
fake_large_original_scales = scaler.inverse_transform(fake_large)
fake_mean = np.mean(fake_large_original_scales, axis=0)
fake_std = np.std(fake_large_original_scales, axis=0)

print("Real data mean:  ", real_mean)
print("Real data std:   ", real_std)
print("Fake data mean:  ", fake_mean)
print("Fake data std:   ", fake_std)

print("M'illumino D'immenso")

# Save the models
torch.save(generator.state_dict(), "ModelsWeights/generator.pth")
torch.save(discriminator.state_dict(), "ModelsWeights/discriminator.pth")
