import numpy as np
import pandas as pd
import os
import time 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoLars
from sklearn.decomposition import FastICA
# from sklearn.preprocessing import normalize
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import random

from sythetic import get_intersections
from independent_stagewise import (
    get_res_y, 
    get_res_y_2, 
    add_res_y, 
    add_res_y_2,
)


seed = 10
random.seed(seed)
np.random.seed(seed)


class MultimodalDataset(Dataset):
  def __init__(self, total_data, total_labels):
    self.data = torch.from_numpy(total_data).type(torch.float32)
    self.labels = torch.from_numpy(total_labels).type(torch.float32)
    self.num_modalities = self.data.shape[1]
  
  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    data = [self.data[idx, i] for i in range(self.num_modalities)] + [self.labels[idx]]
    return data


class EarlyFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.layer1 = nn.Linear(self.input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h = F.relu(self.layer1(x))
        # out = torch.tanh(self.layer2(h))
        out = self.layer2(h)
        return out


class LateFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, output_dim=1):
        super().__init__()
        self.num_modalities = input_dim
        self.encoders = nn.ModuleList([nn.Linear(1, hidden_dim) for i in range(self.num_modalities)])
        self.mlp = nn.Sequential(nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        hs = [F.relu(self.encoders[i](x[:, i:i+1])) for i in range(self.num_modalities)]
        # out = torch.tanh(self.mlp(torch.concat(hs, dim=1)))
        out = self.mlp(torch.concat(hs, dim=1))
        return out


# Model Training
def train_EF(model, iterator, optimizer, criterion):
    epoch_loss = 0

    model.train()

    for i_batch, data_batch in enumerate(iterator):
        x = torch.stack(data_batch[:-1], dim=1).to(device)
        y = data_batch[-1].to(device)

        optimizer.zero_grad()
        y_pred = model(x)

        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Model Evaluation
def eval_EF(model, iterator, criterion):
    epoch_loss = 0

    model.eval()

    with torch.no_grad():
        for i_batch, data_batch in enumerate(iterator):
          x = torch.stack(data_batch[:-1], dim=1).to(device)
          y = data_batch[-1].to(device)

          y_pred = model(x)

          loss = criterion(y_pred, y)

          epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def get_corr(X, y):
    corr, _ = pearsonr(X, y)
    return corr

def normalization(data, normalization_method='uniform', data_type='df'):
    if data_type == 'df':
        if normalization_method == 'uniform':  # into range [0, 1]
            for column in data.columns:
                data[column] = (data[column] -  data[column].min()) / (data[column].max() - data[column].min())
        else:
            for column in data.columns:  # into N(0, 1)
                data[column] = (data[column] -
                                    data[column].mean()) / data[column].std()  
    else:
        if normalization_method == 'uniform':  # into range [0, 1]
            data = (data - np.amin(data, axis=0)) / (np.amax(data, axis=0) - np.amin(data, axis=0))
        else:
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)  
    return data

########################## Load Raw Data ##########################
df = pd.read_csv(os.path.join('~/data/', 'HomeC.csv'), low_memory=False)
print(df.head(), '\n', df.columns)

########################## Data Preprocessing ##########################
# Rename columns to remove spaces and the kW unit 
df.columns = [col[:-5].replace(' ','_') if 'kW' in col else col for col in df.columns]

# Drop rows with nan values 
df = df.dropna()

# The columns "use" and "house_overall" are the same, so let's remove the 'house_overall' column
df.drop(['House_overall'], axis=1, inplace=True)

# The columns "gen" and "solar" are the same, so let's remove the 'solar' column
df.drop(['Solar'], axis=1, inplace=True)

# drop rows with cloudCover column values that are not numeric (bug in sensors) and convert column to numeric
df = df[df['cloudCover']!='cloudCover']
df["cloudCover"] = pd.to_numeric(df["cloudCover"])

# Create columns that regroup kitchens and furnaces 
df['kitchen'] = df['Kitchen_12'] + df['Kitchen_14'] + df['Kitchen_38']
df['Furnace'] = df['Furnace_1'] + df['Furnace_2']

# Convert "time" column (which is a unix timestamp) to a Y-m-d H-M-S 
start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(df['time'].iloc[0])))
time_index = pd.date_range(start_time, periods=len(df), freq='min')  
time_index = pd.DatetimeIndex(time_index)
df = df.set_index(time_index)
df = df.drop(['time'], axis=1)

print(df.shape, '\n', df.columns)

########################## Modeling ##########################
# Task: Predicting future energy consumption by utilizing weather information
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper-parameters
num_modalities = 10
batch_size = 128
criterion = nn.MSELoss()

# Define datasets and labels TODO: transform icon and summary to one-hot
df_weather = df[['temperature',
    #    'icon', 
       'humidity', 'visibility', 
    #    'summary', 
       'apparentTemperature',
       'pressure', 'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity',
       'dewPoint', 'precipProbability', 'use']]
print(df_weather.shape, '\n', df_weather.head(), print(df_weather.dtypes))

# Drop dulicated rows
df_weather = df_weather.drop_duplicates()

# Select modalities
df_weather = df_weather.iloc[:, [i for i in range(num_modalities)] + [-1]]

# Normalization
normalization_method = 'normal'
df_weather = normalization(df_weather, normalization_method, data_type='df') 
print(sum(df_weather.iloc[:, 0]), max(df_weather.iloc[:, 0]), min(df_weather.iloc[:, 0]))
print(df_weather.iloc[:, 0].mean(), df_weather.iloc[:, 0].std)

y_total = df_weather[['use']].values.ravel()
X_total = df_weather.drop(columns=['use']).values
num_data = X_total.shape[0]

# Train
print(f"Start Training...")
model_list = []
test_score_list = []
corr_list = []

# Wrap dataloader
dataset = MultimodalDataset(X_total, y_total)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8*num_data), num_data-int(0.8*num_data)])

train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True,
                        batch_size=batch_size,
                        num_workers=4)
test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                        batch_size=batch_size,
                        num_workers=4)


early_fusion_losses, late_fusion_losses = [], []
for _ in range(3):  # run 3 times
    # Early Fusion
    model = EarlyFusion(num_modalities)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 30
    for i in range(num_epochs):
        train_loss = train_EF(model, train_loader, optimizer, criterion)
        test_loss = eval_EF(model, test_loader, criterion)
        print(f"***Epoch {i}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}")
    early_fusion_losses.append(test_loss)

    # Late Fusion
    model = LateFusion(num_modalities)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 30
    for i in range(num_epochs):
        train_loss = train_EF(model, train_loader, optimizer, criterion)
        test_loss = eval_EF(model, test_loader, criterion)
        print(f"***Epoch {i}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}")
    late_fusion_losses.append(test_loss)

print(f"Finished Training!\n ***Early fusion: {early_fusion_losses}\n ***Late fusion: {late_fusion_losses}")
print(f"***Mean of Early Fusion: {sum(early_fusion_losses)/len(early_fusion_losses)}\n ***Mean of Late Fusion: {sum(late_fusion_losses)/len(late_fusion_losses)}")