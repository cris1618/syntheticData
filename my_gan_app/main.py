from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse
import uvicorn
import io
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from fastapi.middleware.cors import CORSMiddleware

# API call
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
    # add any other URLs if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

            # Train discriminator
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # real
            d_real = discriminator(real_data)
            d_real_loss = criterion(d_real, real_labels)

            # fake
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            d_fake = discriminator(fake_data.detach())
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            d_fake_for_g = discriminator(fake_data)
            g_loss = criterion(d_fake_for_g, real_labels)
            g_loss.backward()
            optimizer_G.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}"
                  f"| Discriminator Loss: {d_loss.item():.4f},"
                  f"Generator Loss: {g_loss.item():.4f}")
    
    return generator

# Actual endpoint of the API
@app.post("/generate_synthetic")
async def generate_synthetic(
    file: UploadFile = File(...),
    num_rows: int = 50,
    epochs: int = 50
) -> PlainTextResponse:
    """
    1) Read the CSV from the uploaded file
    2) Preprocess the data
    3) Retrain the GAN
    4) Generate "num_rows" of synthetic data
    5) Return CSV text as the response
    """
    
    # Load user CSV into a dataframe
    content = await file.read() # read raw bytes
    df = pd.read_csv(io.BytesIO(content))

    # Basic check
    if len(df) < 10:
        return PlainTextResponse("Dataset too small, try with more rows.", status_code=400)
    
    # Preprocess the data
    # Separate numeric and categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Clean possible missing values
    df.dropna(how="all", axis=1, inplace=True)

    # build a ColumnTransformer for full preprocessing
    # - Numeric: impute mean, scale
    # - Categorical: impute mode, one-hot encoding

    numeric_transformer = Pipeline([
        ("num_imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop"
    )

    data_array = preprocessor.fit_transform(df)
    if hasattr(data_array, "toarray"):
        data_array = data_array.toarray()

    if data_array.shape[1] == 0:
        return PlainTextResponse("No valid columns found after preprocessing!", status_code=400)

    # If too few rows remain, also handle
    if data_array.shape[0] < 10:
        return PlainTextResponse("Not enough rows to train a GAN!", status_code=400)

    # Train the GAN on the data
    generator = train_gan_on_data(data_array, num_rows, epochs)

    # Generate the data
    latent_dim = 64
    z = torch.randn(num_rows, latent_dim)
    synthetic_scaled  = generator(z).detach().numpy()

    # retransform data to original version
    num_count = len(numeric_cols)
    cat_count_ohe = synthetic_scaled.shape[1] - num_count

    X_num_fake = synthetic_scaled[:, :num_count]
    X_cat_fake = synthetic_scaled[:, num_count:]

    # inverse transform
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    X_num_original_scale = scaler.inverse_transform(X_num_fake)

    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    X_cat_original_labels = encoder.inverse_transform(X_cat_fake)

    # Reconstruct the dataset
    df_num = pd.DataFrame(X_num_original_scale, columns=numeric_cols)
    df_cat = pd.DataFrame(X_cat_original_labels, columns=cat_cols)

    synthetic_df = pd.concat([df_num, df_cat], axis=1)

    # put the columns in original order
    original_order = list(df.columns)
    synthetic_df = synthetic_df[original_order]

    # Create the CSV to return 
    csv_buffer = io.StringIO()
    synthetic_df.to_csv(csv_buffer, index=False)
    csv_text = csv_buffer.getvalue()

    return PlainTextResponse(csv_text, media_type="text/csv")

        



