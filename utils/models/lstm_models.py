from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
from transformers import pipeline


class AttentionLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            dropout=dropout if num_layers > 1 else 0,  # Dropout только между слоями
            batch_first=True
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)  # Дополнительный dropout
        self.bn = nn.BatchNorm1d(hidden_size)  # BatchNorm
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out, _ = self.attention(out, out, out)  # Механизм внимания
        out = out[:, -1, :]  # берём последний шаг
        out = self.bn(out)
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)
    
class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            dropout=0.3 if num_layers > 1 else 0,  # Dropout только между слоями
            batch_first=True
        )
        self.dropout = nn.Dropout(0.5)  # Дополнительный dropout
        self.bn = nn.BatchNorm1d(hidden_size)  # BatchNorm
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        self.sigmoid = nn.Sigmoid()
        
        # Инициализация весов
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        # Инициализация скрытых состояний
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # берём последний шаг
        out = self.bn(out)
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)
    

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            dropout=0.5 if num_layers > 1 else 0,  # Dropout только между слоями
            batch_first=True
        )
        self.dropout = nn.Dropout(0.5)  # Дополнительный dropout
        self.bn = nn.BatchNorm1d(hidden_size)  # BatchNorm
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )
        
        # Инициализация весов
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        # Инициализация скрытых состояний
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # берём последний шаг
        out = self.bn(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out
    
class AttentionLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            dropout=dropout if num_layers > 1 else 0,  # Dropout только между слоями
            batch_first=True
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)  # Дополнительный dropout
        self.bn = nn.BatchNorm1d(hidden_size)  # BatchNorm
        self.fc = nn.Linear(hidden_size, 1)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out, _ = self.attention(out, out, out)  # Механизм внимания
        out = out[:, -1, :]  # берём последний шаг
        out = self.bn(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


    
class SentimentModel(nn.Module):
    def __init__(self, model='seara/rubert-tiny2-russian-sentiment'):
        super().__init__()
        self.model = pipeline("sentiment-analysis", model=model)

    def forward(self, x):
        with torch.no_grad():
            out = self.model(x)
        return out