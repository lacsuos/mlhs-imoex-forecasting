from torch.utils.data import Dataset, DataLoader
from torch import nn
import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DatasetCreator:
    def __init__(self, window_size=30, scaler=MinMaxScaler(feature_range=(-1, 1)), train=False):
        self.window_size=window_size
        self.scaler = scaler
        self.train = train

    def __create_sequences(self, data, features, target, date_col='date'):
        X, y = [], []
        if self.train: 
            for i in range(len(data) - self.window_size):
                X.append(data[features].values[i:i+self.window_size])
                y.append(data[target].values[i+self.window_size])
            return torch.FloatTensor(X, device=DEVICE), torch.FloatTensor(y, device=DEVICE).view(-1, 1)
        else:
            for i in range(len(data) - self.window_size):
                X.append(data[features].values[i:i+self.window_size])
            return np.array(X)
            
    def create_datasets(self, data, features, target, train_size=0.9):
        if self.train:
            data=data.copy()
            data[features] = self.scaler.fit_transform(data[features])
            X, y = self.__create_sequences(data, features, target)
            print('sequences created')
            train_size = int(train_size * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Конвертация в тензоры PyTorch
            print('transforming to tensors')
            print('creating datasets')
            train_dataset = TimeSeriesDataset(X_train, y_train)
            test_dataset = TimeSeriesDataset(X_test, y_test)              
            return train_dataset, test_dataset
        else:
            data[features] = self.scaler.transform(data[features])
            X = self.__create_sequences(data, features, target)
            X = torch.FloatTensor(X)
            return X

