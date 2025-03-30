from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import uvicorn
import io
import numpy as np
import pandas as pd
from ctgan import CTGAN

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate_synthetic")
async def generate_synthetic(
    file: UploadFile = File(...), 
    num_rows: int = Form(...),
    categorical_columns: str = Form("")
) -> PlainTextResponse:
    """
    Generate synthetic CSV using CTGAN. 

    Steps:
      1) Read the CSV from the uploaded file 
      2) Detect columns to treat as discrete vs. continuous
      3) Train CTGAN on the raw dataframe
      4) Generate synthetic data
      5) Return the CSV.
    """
    # Read CSV from user
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    df.columns = df.columns.str.strip()

    # Parse datetime columns
    datetime_columns = []
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if converted.notna().mean() > 0.8:
                converted = converted.fillna(method="ffill")
                # Convert to Unix timestamp (seconds)
                df[col] = converted.astype(np.int64) // 10**9
                datetime_columns.append(col)
                print(f"Column '{col}' was parsed as datetime -> converted to Unix timestamp.")

    # Enforce numeric for everything recognized or newly numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Quick check: Must have enough data
    if len(df) < 10:
        return PlainTextResponse("Dataset too small, please provide more rows.", status_code=400)

    # Build initial list of discrete columns by type
    discrete_columns = list(df.select_dtypes(include=["object", "category", "bool"]).columns)

    # If user explicitly provided some columns to treat as categorical
    if categorical_columns:
        specified_cat_cols = [c.strip() for c in categorical_columns.split(",")]
        for c in specified_cat_cols:
            if c in df.columns:
                # Force these as strings
                df[c] = df[c].astype(str)
        discrete_columns.extend(specified_cat_cols)

    """# Next, treat numeric columns with few unique values as categorical
    for col in numeric_cols:
        unique_vals = df[col].nunique(dropna=False)
        if unique_vals <= 10:
            # This is presumably a "category" disguised as numeric
            df[col] = df[col].astype(str)
            discrete_columns.append(col)
            print(f"Column '{col}' has <=10 unique values; treating as categorical.")
    """
    # Clean the discrete_columns set, removing any datetime col
    discrete_columns = list(set(discrete_columns) - set(datetime_columns))

    # Check for mixed types in columns
    for col in df.columns:
        sample_types = set(df[col].dropna().map(type))
        if len(sample_types) > 1:
            print(f"Warning: Column '{col}' has mixed types {sample_types}. Removing it.")
            df.drop(columns=[col], inplace=True)

    # If everything ended up dropped
    if df.shape[1] == 0:
        return PlainTextResponse("No valid columns left after cleaning. Check input data.", status_code=400)

    # Now see what's final
    discrete_columns = [c for c in discrete_columns if c in df.columns]

    # Drop columns that are completely empty
    df.dropna(how="all", axis=1, inplace=True)

    # Train CTGAN
    model = CTGAN(epochs=100)
    model.fit(df, discrete_columns=discrete_columns)

    # Generate synthetic data
    synthetic_df = model.sample(num_rows)

    #  Convert timestamps back to datetime if desired
    for col in datetime_columns:
        if col in synthetic_df.columns:
            synthetic_df[col] = pd.to_datetime(synthetic_df[col], unit="s", origin="unix")

    # Replace string "nan" with np.nan for all columns
    synthetic_df.replace("nan", np.nan, inplace=True)
    
    # Separate numeric and categorical
    num_cols = synthetic_df.select_dtypes(include=[np.number]).columns
    cat_cols = synthetic_df.select_dtypes(exclude=[np.number]).columns

    # Fill numeric with mean
    for col in num_cols:
        col_mean = synthetic_df[col].mean(skipna=True)
        synthetic_df[col].fillna(col_mean, inplace=True)
    
    # Fill categorical with most frequent
    for col in cat_cols:
        mode_val = synthetic_df[col].mode(dropna=True)
        if not mode_val.empty:
            synthetic_df[col].fillna(mode_val[0], inplace=True)
        else:
            synthetic_df[col].fillna("MISSING", inplace=True)

    # Return CSV
    csv_buf = io.StringIO()
    synthetic_df.to_csv(csv_buf, index=False)
    csv_text = csv_buf.getvalue()

    return PlainTextResponse(csv_text, media_type="text/csv")
