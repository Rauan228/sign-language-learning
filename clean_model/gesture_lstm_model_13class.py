# -*- coding: utf-8 -*-
"""
LSTM модель для распознавания динамических жестов (13 классов)
Использует последовательности ключевых точек рук для классификации
Адаптирована из clean_model/gesture_lstm_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class GestureLSTM(nn.Module):
    def __init__(self, input_size=126, hidden_size=128, num_layers=2, num_classes=13, dropout=0.3):
        super(GestureLSTM, self).__init__()
        
        self.input_size = input_size  # 126 признаков (2 руки * 21 точка * 3 координаты)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Входной слой для нормализации
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention механизм
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Классификационные слои
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов модели"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Забываем гейт bias = 1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
    
    def forward(self, x, lengths=None):
        """
        Forward pass
        Args:
            x: (batch_size, seq_len, input_size)
            lengths: (batch_size,) - длины последовательностей
        """
        batch_size, seq_len, _ = x.size()
        
        # Нормализация входных данных
        x_reshaped = x.view(-1, self.input_size)
        x_norm = self.input_norm(x_reshaped)
        x = x_norm.view(batch_size, seq_len, self.input_size)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention механизм
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Глобальное усреднение по времени с учетом attention
        if lengths is not None:
            # Маскирование для переменной длины
            mask = torch.arange(seq_len).expand(batch_size, seq_len).to(x.device)
            mask = mask < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            
            attn_out = attn_out * mask
            pooled = attn_out.sum(dim=1) / lengths.unsqueeze(1).float()
        else:
            # Простое усреднение
            pooled = attn_out.mean(dim=1)
        
        # Классификация
        output = self.classifier(pooled)
        
        return output
    
    def get_attention_weights(self, x):
        """Получение весов attention для визуализации"""
        batch_size, seq_len, _ = x.size()
        
        # Нормализация
        x_reshaped = x.view(-1, self.input_size)
        x_norm = self.input_norm(x_reshaped)
        x = x_norm.view(batch_size, seq_len, self.input_size)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        _, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        return attn_weights

class TemporalConvNet(nn.Module):
    """Альтернативная архитектура с временными свертками"""
    def __init__(self, input_size=126, num_classes=13, num_channels=[64, 128, 256], kernel_size=3, dropout=0.3):
        super(TemporalConvNet, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Входная нормализация
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Временные сверточные слои
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                  stride=1, dilation=dilation_size,
                                  padding=(kernel_size-1) * dilation_size // 2))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        
        # Глобальное усреднение и классификация
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.size()
        
        # Перестановка для Conv1d: (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # Нормализация
        x_norm = self.input_norm(x)
        
        # TCN
        tcn_out = self.tcn(x_norm)
        
        # Глобальное усреднение
        pooled = self.global_pool(tcn_out).squeeze(-1)
        
        # Классификация
        output = self.classifier(pooled)
        
        return output

def create_model(model_type='lstm', **kwargs):
    """Фабрика для создания моделей"""
    if model_type == 'lstm':
        return GestureLSTM(**kwargs)
    elif model_type == 'tcn':
        return TemporalConvNet(**kwargs)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

if __name__ == '__main__':
    # Тестирование модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создание тестовых данных
    batch_size = 4
    seq_len = 30
    input_size = 126
    num_classes = 13
    
    x = torch.randn(batch_size, seq_len, input_size).to(device)
    lengths = torch.tensor([30, 25, 20, 15]).to(device)
    
    # Тестирование LSTM модели
    print("Тестирование LSTM модели...")
    lstm_model = GestureLSTM(input_size=input_size, num_classes=num_classes).to(device)
    
    with torch.no_grad():
        output = lstm_model(x, lengths)
        print(f"Выход LSTM: {output.shape}")
        
        # Тестирование attention весов
        attn_weights = lstm_model.get_attention_weights(x)
        print(f"Attention веса: {attn_weights.shape}")
    
    # Тестирование TCN модели
    print("\nТестирование TCN модели...")
    tcn_model = TemporalConvNet(input_size=input_size, num_classes=num_classes).to(device)
    
    with torch.no_grad():
        output = tcn_model(x)
        print(f"Выход TCN: {output.shape}")
    
    # Подсчет параметров
    lstm_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
    tcn_params = sum(p.numel() for p in tcn_model.parameters() if p.requires_grad)
    
    print(f"\nПараметры LSTM: {lstm_params:,}")
    print(f"Параметры TCN: {tcn_params:,}")
    
    print("\nТестирование завершено успешно!")