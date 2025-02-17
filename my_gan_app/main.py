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