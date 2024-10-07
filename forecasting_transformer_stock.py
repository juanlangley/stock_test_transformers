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

data = yf.download('GBPJPY=x', interval="1m",)

def preprocessing_yf(data):
  #Importar los datos
  df = data.dropna()
  df.columns = ["Open", "High", "Low", "Close", "Adj close", "Volume"]
  # Eliminar la columna adj close
  del df["Adj close"]
  return df

data = preprocessing_yf(data)
features = feature_engineering(data).dropna()

adx_array = features["ADX_14"].to_numpy()
adx_array = data["Close"].to_numpy()[2500:5000]
adx_array = features["SMA50"].to_numpy()

# A basic attention mechanism
class Attention_basic(nn.Module):
    def __init__(self, seq_len=200, device="cuda"):
        super(Attention_basic, self).__init__()
        self.device=device
        self.queries = nn.Linear(seq_len, seq_len)
        self.keys = nn.Linear(seq_len, seq_len)
        self.values = nn.Linear(seq_len, seq_len)
    def forward(self, x, mask=True):
        q = self.queries(x).reshape(x.shape[0], x.shape[1], 1)
        k = self.keys(x).reshape(x.shape[0], x.shape[1], 1)
        v = self.values(x).reshape(x.shape[0], x.shape[1], 1)
        scores = torch.bmm(q, k.transpose(-2, -1))
        if mask:
            maskmat = torch.tril(torch.ones((x.shape[1], x.shape[1]))).to(self.device)
            scores = scores.masked_fill(maskmat == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.bmm(attention_weights, v)
        return output.reshape(output.shape[0], output.shape[1])
    
# A forcasting model
class ForecastingModel(torch.nn.Module):
    def __init__(self, seq_len=200, ffdim=64, device="cuda"):
        super(ForecastingModel, self).__init__()
        self.relu = nn.ReLU()
        self.attention = Attention_basic(seq_len, device=device)
        self.linear1 = nn.Linear(seq_len, int(ffdim))
        self.linear2 = nn.Linear(int(ffdim), int(ffdim/2))
        self.linear3 = nn.Linear(int(ffdim/2), int(ffdim/4))
        self.outlayer = nn.Linear(int(ffdim/4), 1)
    def forward(self, x):
        x = self.attention(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        return self.outlayer(x)
    


# Create a dataset
DATA_SIZE = 3000
seq_len = 500
nw_array = adx_array[0:DATA_SIZE]
X = np.array([nw_array[ii:ii+seq_len] for ii in range(0, nw_array.shape[0]-seq_len)])
Y = np.array([nw_array[ii+seq_len] for ii in range(0, nw_array.shape[0]-seq_len)])

device = "cuda" if torch.cuda.is_available() else "cpu"
# Training Loop
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 4.12e-5
model = ForecastingModel(seq_len, device = device)
model.train()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
dataset = TensorDataset(torch.Tensor(X).to(device), torch.Tensor(Y).to(device))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
for epoch in range(EPOCHS):
    for xx, yy in dataloader:
        optimizer.zero_grad()
        out = model(xx)
        loss = criterion(out, yy)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={loss}")

# New Prediction Loop
FORCAST = 1000
model.eval()
for ff in range(FORCAST):
    xx = nw_array[len(nw_array)-seq_len:len(nw_array)]
    yy = model(torch.Tensor(xx).reshape(1, xx.shape[0]).to(device))
    nw_array = np.concatenate((nw_array, yy.detach().cpu().numpy().reshape(1,)))

# Plot Predictions
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6, 6))
plt.plot(range(nw_array[:DATA_SIZE].shape[0]), nw_array[:DATA_SIZE])
plt.plot(range(nw_array[:DATA_SIZE].shape[0], nw_array.shape[0]), nw_array[DATA_SIZE:DATA_SIZE+FORCAST], 'r--')
plt.plot(range(nw_array[:DATA_SIZE].shape[0], nw_array.shape[0]), adx_array[DATA_SIZE:DATA_SIZE+FORCAST], 'g-')
plt.xlim(2500,3500)
plt.show()