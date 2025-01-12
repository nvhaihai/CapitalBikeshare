from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from .time_feature import time_features



def load_data(data_path):
    data = pd.read_csv(data_path)
    data = data.drop(['instant', 'dteday', 'casual', 'registered'], axis = 1)

    for col in data.columns:
        if not np.issubdtype(data[col].dtype, np.number):
            print(f"列 {col} 包含非数值数据，正在尝试转换...")
            data[col] = pd.to_numeric(data[col], errors='coerce')

    if data.isnull().values.any():
        print("数据中存在 NaN 值，正在删除...")
        data = data.dropna()

    if np.isinf(data.values).any():
        print("数据中存在 Inf 值，正在处理...")
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError("清洗后数据为空，请检查数据来源！")



    return data

def create_dataset(data, input_window, output_window, step=1):
    X, y = [], []
    for i in range(0, len(data) - input_window - output_window + 1, step):
        X.append(data[i:i+input_window])
        y.append(data[i+input_window:i+input_window+output_window])
    return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y[:, :, -1], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TimeSeriesDataset_RevDimTSFormer(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X[:, :, 4:], dtype=torch.float32)
        self.X_mark = torch.tensor(X[:, :, :4], dtype=torch.float32)
        self.X_dec = torch.tensor(y[:, :, 4:-1], dtype=torch.float32)
        self.X_dec_mark = torch.tensor(y[:, :, :4], dtype=torch.float32)

        self.y = torch.tensor(y[:, :, -1], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.X_mark[idx], self.X_dec[idx], self.X_dec_mark[idx], self.y[idx]
