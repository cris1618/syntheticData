from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse
import uvicorn
import io
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

# API call
app = FastAPI()

# Define the model (Generator and discriminator)
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid() # Return probability between 0 and 1
        )
    
    def forward(self, x):
        return self.model(x)

# Class for CSV datasets import
class CSVDataset(Dataset):
    def __init__(self, array):
        super().__init__()
        self.data = torch.tensor(array, dtype=torch.float32)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]

# define a training function for the GAN
def train_gan_on_data(data_array, num_epochs=50, batch_size=16, latent_dim=64, lr=2e-4):
    # Create the dataloader
    dataset = CSVDataset(data_array)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models
    input_dim = data_array.shape[1]
    generator = Generator(latent_dim, input_dim)
    discriminator = Discriminator(input_dim)

    # Loss and optim
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training 
    for epoch in range(num_epochs):
        for i, real_data in enumerate(loader):
            batch_size = real_data.size(0)



