import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import pandas_ta as pdta
import yfinance as yf
import pandas as pd
from ta import trend as trd

def feature_engineering(dataframe):
    df_copy = dataframe.copy()
    df_copy = pd.DataFrame()
    df_copy["SMA50"] = trd.sma_indicator(dataframe["Close"], window=50)
    # ------------- ADX 14 --------------
    adx = pdta.adx(high=dataframe["High"], low=dataframe["Low"], close=dataframe["Close"], length=14)
    df_copy["ADX_14"] = adx["ADX_14"]
    df_copy["DMP_14"] = adx["DMP_14"]
    df_copy["DMN_14"] = adx["DMN_14"]
    df_copy["ADXR_14"] = (adx["ADX_14"] + adx["ADX_14"].shift(14))/2
    return df_copy

data = yf.download('GBPUSD=x', interval="5m",start="2024-09-01")

def preprocessing_yf(data):
  df = data.dropna()
  df.columns = ["Open", "High", "Low", "Close", "Adj close", "Volume"]
  del df["Adj close"]
  return df

data = preprocessing_yf(data)
features = feature_engineering(data)
features = features.dropna()
closes = data["Close"].to_numpy()
feat = features.loc[:,["ADX_14", "DMP_14", "DMN_14", "ADXR_14"]].to_numpy()
shapes = closes.reshape(-1, 1)
print(shapes.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_fit = scaler.fit(shapes)
scaled_data = scaler.transform(shapes)


class Encoder(nn.Module):
    def __init__(self, z_dim=3):
        super(Encoder, self).__init__()
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)

        # Biuld V part of VAE
        self.mu = nn.Linear(2*2*256, self.z_dim)
        self.logvar = nn.Linear(2*2*256, self.z_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(h.size(0), -1)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_dim=3):
        super(Decoder, self).__init__()
        self.z_dim = z_dim

        # Decoder
        self.linear1 = nn.Linear(self.z_dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2)

    def forward(self, z):
        h = F.relu(self.linear1(z))
        h = h.view(h.size(0), 1024, 1, 1)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        y = torch.sigmoid(self.deconv4(h))
        return y

class ConvVAE(nn.Module):
    def __init__(self, z_dim=32):
        super(ConvVAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(logvar / 2)
        epsilon = torch.randn(mu.size(0), mu.size(1))
        z = mu + sigma * epsilon
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        y = self.decoder(z)
        return y, mu, logvar

z_size = 128  # Define el tamaño de z
batch_size = 10  # Define el tamaño del lote
model = ConvVAE(z_dim=z_size)
# Supongamos que tienes una entrada de ejemplo
x = torch.randn(batch_size, 3, 64, 64)  # Batch size 10, 3 canales, 64x64 imágenes

# Pasar la entrada a través del modelo
y, mu, logvar = model(x)
print(y.shape)  # Debe ser [batch_size, 3, 64, 64]
print(mu.shape)
print(logvar.shape)