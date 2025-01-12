import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.load_data import load_data, create_dataset, TimeSeriesDataset, TimeSeriesDataset_RevDimTSFormer
from models import LSTMModel, TransformerModel, RevDimTSFormerModel
from sklearn.preprocessing import MinMaxScaler
from utils.train import train_model, train_RevDimTSFormer
from utils.result_plot import predict_and_plot
from torch.utils.data import DataLoader
import os
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.evaluation import mse_mae, mse_mae_RevDimTSFormer
import warnings

class Configs:
    def __init__(self):
        self.seq_len = 96
        self.pred_len = 96
        self.d_model = 64
        self.embed = 'fixed'
        self.freq = 'D'
        self.dropout = 0.1
        self.output_attention = False
        self.use_norm = True
        self.class_strategy = 'classification'
        self.e_layers = 4
        self.n_heads = 8
        self.d_ff = 256
        self.activation = 'relu'
        self.factor = 1

def main(model_name, input_window, output_window, step, exp_id):
    train_data_path = 'dataset/train_data.csv'
    test_data_path = 'dataset/test_data.csv'
    model_path = f'results/{model_name}_out{output_window}_step{step}_{exp_id}.pth'

    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)
    input_size = train_data.shape[1]

    scaler_all = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler_all.fit_transform(train_data)
    scaled_test_data = scaler_all.transform(test_data)

    scaler_cnt = MinMaxScaler(feature_range=(0, 1))
    scaled_cnt = scaler_cnt.fit_transform(train_data.iloc[:, -1].to_frame())
    scaled_test_cnt = scaler_cnt.transform(test_data.iloc[:, -1].to_frame())


    if np.isnan(scaled_values).any() or np.isinf(scaled_values).any():
        print("数据包含 NaN 或 Inf，请检查数据预处理步骤！")
        exit()

    X, y = create_dataset(scaled_values, input_window, output_window, step=step)
    X_test, y_test = create_dataset(scaled_test_data, input_window, output_window, step=1)     # test集的数据加载步长始终为1

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=42)     # 切分成train-dev集，发现使用train_test_split进行数据切分的训练效果更好
    # train_size = int(len(X)*0.8)
    # X_train, X_dev, y_train, y_dev = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    if model_name == 'RevDimTSFormer':
        train_dataset = TimeSeriesDataset_RevDimTSFormer(X_train, y_train)
        dev_dataset = TimeSeriesDataset_RevDimTSFormer(X_dev, y_dev)
        test_dataset = TimeSeriesDataset_RevDimTSFormer(X_test, y_test)
    else:
        train_dataset = TimeSeriesDataset(X_train, y_train)
        dev_dataset = TimeSeriesDataset(X_dev, y_dev)
        test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    print(model_name)

    if model_name == 'LSTM':
        model = LSTMModel(input_size=input_size, hidden_size=64, output_steps=output_window)
    elif model_name == 'Transformer':
        model = TransformerModel(input_dim=input_size, model_dim=64, num_heads=8, num_layers=4, output_dim=output_window)
    elif model_name == 'RevDimTSFormer':
        configs = Configs()
        configs.seq_len = input_window
        configs.pred_len = output_window
        model = RevDimTSFormerModel(configs)

    print("Training Model")
    if model_name == 'RevDimTSFormer':
        train_RevDimTSFormer(model, train_loader, dev_loader, epochs=500, lr=0.001, device=device, model_path=model_path)
        warnings.filterwarnings("ignore", category = FutureWarning)
        model.load_state_dict(torch.load(model_path))
        mse_mae_RevDimTSFormer(model, test_loader, scaler_cnt, device, model_name, output_window, exp_id)
    else:
        train_model(model, train_loader, dev_loader, epochs=500, lr=0.001, device=device, model_path=model_path)
        warnings.filterwarnings("ignore", category = FutureWarning)
        model.load_state_dict(torch.load(model_path))
        mse_mae(model, test_loader, scaler_cnt, device, model_name, output_window, exp_id)

    


if __name__=='__main__':
    model_name = 'Transformer'
    input_window = 96
    output_window = 96
    step = 1
    exp_id = "1"

    main(model_name, input_window, output_window, step, exp_id)