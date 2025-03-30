from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import PlainTextResponse
import uvicorn
import io
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    "https://syntheticdata-production.up.railway.app" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains (you can restrict it later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the model (Generator and discriminator)
# Using a Conditional GAN to handle discrete variables
"""class cGenerator(nn.Module):
    def __init__(self, latent_dim, cond_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, z, c):
        # z = [batch_size, latent_dim]
        # c = [batch_size, cond_dim]
        x = torch.cat([z,c], dim=1)
        return self.model(x)"""

class cGenerator(nn.Module):
    def __init__(self, latent_dim, cond_dim, numeric_dim, cat_dims):
        super().__init__()
        self.numeric_dim = numeric_dim
        self.cat_dims = cat_dims = cat_dims
        total_cat_dim = sum(cat_dims)
        total_output_dim = numeric_dim + total_cat_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, total_output_dim)
        )
        self.temperature = 0.1
    
    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        out = self.fc(x)
        numeric_out = out[:, :self.numeric_dim]
        cat_logits = out[:, self.numeric_dim:]
        cat_outputs = []
        start = 0
        # Apply Gumbel-Softmax
        for dim in self.cat_dims:
            logits = cat_logits[:, start:start+dim]
            one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
            cat_outputs.append(one_hot)
            start += dim
        cat_out = torch.cat(cat_outputs, dim=1)
        return torch.cat([numeric_out, cat_out], dim=1)


class cDiscriminator(nn.Module):
    def __init__(self, input_dim, cond_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid() # Return probability between 0 and 1
        )
    
    def forward(self, x, c):
        merged = torch.cat([x, c], dim=1)
        return self.model(merged)

# Class for CSV datasets import
class cGAN_Datset(Dataset):
    def __init__(self, data, cond):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.cond = torch.tensor(cond, dtype=torch.float32)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.cond[idx]

# define a training function for the GAN
def train_cgan_on_data(data_array, 
                       cond_array,
                       numeric_dim,
                       cat_dims,
                       num_epochs=50, 
                       batch_size=16, 
                       latent_dim=64, 
                       lr=2e-4):
    # Create the dataloader
    dataset = cGAN_Datset(data_array, cond_array)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models
    input_dim = data_array.shape[1]
    cond_dim = cond_array.shape[1]

    generator = cGenerator(latent_dim, cond_dim, numeric_dim, cat_dims)
    discriminator = cDiscriminator(input_dim, cond_dim)

    # Loss and optim
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training 
    for epoch in range(num_epochs):
        for i, (real_data, real_cond) in enumerate(loader):
            batch_size = real_data.size(0)

            # Labels
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Train discriminator
            optimizer_D.zero_grad()
            
            # real
            d_real = discriminator(real_data, real_cond)
            d_real_loss = criterion(d_real, real_labels)

            # fake
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z, real_cond)
            d_fake = discriminator(fake_data.detach(), real_cond)
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z, real_cond)
            d_fake_for_g = discriminator(fake_data, real_cond)
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
    num_rows: int = Form(50),
    #epochs: int = Form(50)
    categorical_columns: str = Form(""),
    categorical_mapppings: str = Form("")
) -> PlainTextResponse:
    """
    1) Read the CSV from the uploaded file
    2) Preprocess the data
    3) Split the preprocessed data
    4) Retrain the GAN
    5) Generate "num_rows" of synthetic data
    6) Inverse transform the data
    7) Return CSV text as the response
    """
    
    # Load user CSV into a dataframe
    content = await file.read() # read raw bytes
    df = pd.read_csv(io.BytesIO(content))

   # Basic check for minimum number of rows
    if len(df) < 10:
        return PlainTextResponse("Dataset too small, try with more rows.", status_code=400)
    
    if categorical_columns:
        specified_cat_cols = [col.strip() for col in categorical_columns.split(",")]
        for col in specified_cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("object")

    # Define default categorical columns (based on data types)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    default_cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

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
            ("cat", categorical_transformer, default_cat_cols)
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

    # Split the preprocessed data into numeric and categorical
    num_count = len(numeric_cols)
    numeric_data = data_array[:, :num_count]
    condition_data = data_array[:, num_count:]

    # Get categorical dimensions from OneHotEncoder
    encoder_instance = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_dims = [len(cat) for cat in encoder_instance.categories_]

    # Extract categorical mappings
    mapping_dict = {}
    if categorical_mapppings:
        mappings = [m.strip() for m in categorical_mapppings.split(";") if m.strip()]
        for m in mappings:
            if ":" in m:
                col, classes_str = m.split(":", 1)
                mapping_dict[col.strip()] = [cls.strip() for cls in classes_str.split(",")]

    # Train the GAN on the data
    num_epochs = 100
    generator = train_cgan_on_data(data_array, condition_data, num_count, cat_dims, num_epochs)

    # Generate the data
    latent_dim = 64
    z = torch.randn(num_rows, latent_dim)
    idx = torch.randint(0, condition_data.shape[0], (num_rows,))
    random_cond = torch.tensor(condition_data[idx], dtype=torch.float32)

    synthetic_full = generator(z, random_cond).detach().numpy()
    synthetic_numeric = synthetic_full[:, :num_count]
    synthetic_cat = synthetic_full[:, num_count:]

    # Inverse transform
    scaler_instance = preprocessor.named_transformers_["num"].named_steps["scaler"]
    X_num_original_scale = scaler_instance.inverse_transform(synthetic_numeric)

    # Process category with the one given by the user
    def  map_column(col_segment, valid_classes):
        indices = col_segment.argmax(axis=1)
        return np.array([valid_classes[idx] for idx in indices]).reshape(-1,1)

    if synthetic_cat.shape[1] > 0:
        mapped_columns = []
        start = 0
        for i, col in enumerate(default_cat_cols):
            dim = cat_dims[i]
            col_segment = synthetic_cat[:, start:start+dim]
            start += dim
            if col in mapping_dict:
                mapped_col = map_column(col_segment, mapping_dict[col])
            else:
                mapped_col = map_column(col_segment, encoder_instance.categories_[i])
            mapped_columns.append(mapped_col)
        X_cat_original_labels = np.hstack(mapped_columns)
    else:
        X_cat_original_labels = np.empty((num_rows, 0))
            

    # Reconstruct the final dataset.
    df_num = pd.DataFrame(X_num_original_scale, columns=numeric_cols)
    if X_cat_original_labels.size > 0:
        df_cat = pd.DataFrame(X_cat_original_labels, columns=default_cat_cols)
        synthetic_df = pd.concat([df_num, df_cat], axis=1)
        original_order = numeric_cols + default_cat_cols
        synthetic_df = synthetic_df[original_order]
    else:
        synthetic_df = df_num

    # Create the CSV to return
    csv_buffer = io.StringIO()
    synthetic_df.to_csv(csv_buffer, index=False)
    csv_text = csv_buffer.getvalue()

    return PlainTextResponse(csv_text, media_type="text/csv")

"""if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not found
    uvicorn.run(app, host="0.0.0.0", port=port)
    #port = int(os.environ.get("PORT", 8000))  # Use Railway's assigned PORT or default to 8000
    #uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)"""

        



