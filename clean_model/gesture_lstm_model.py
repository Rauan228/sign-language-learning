# -*- coding: utf-8 -*-
"""
LSTM модель для распознавания динамических жестов
Использует последовательности ключевых точек рук для классификации
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class GestureLSTM(nn.Module):
    def __init__(self, input_size=84, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super(GestureLSTM, self).__init__()
        
        self.input_size = input_size  # 84 признака (2 руки * 21 точка * 2 координаты)
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
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                
    def forward(self, x, lengths=None):
        """
        Прямой проход модели
        x: (batch_size, seq_len, input_size)
        lengths: длины последовательностей для каждого образца
        """
        batch_size, seq_len, _ = x.shape
        
        # Нормализация входных данных
        x_reshaped = x.view(-1, self.input_size)
        x_norm = self.input_norm(x_reshaped)
        x = x_norm.view(batch_size, seq_len, self.input_size)
        
        # LSTM обработка
        if lengths is not None:
            # Упаковка последовательностей для эффективной обработки
            # lengths должен быть на CPU для pack_padded_sequence
            lengths_cpu = lengths.cpu() if lengths.is_cuda else lengths
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, (hidden, cell) = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True
            )
        else:
            lstm_out, (hidden, cell) = self.lstm(x)
            
        # Attention механизм
        attended_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Глобальное усреднение с учетом длин последовательностей
        if lengths is not None:
            mask = torch.arange(attended_out.size(1)).expand(
                len(lengths), attended_out.size(1)
            ).to(attended_out.device) < lengths.unsqueeze(1)
            
            masked_out = attended_out * mask.unsqueeze(-1).float()
            pooled = masked_out.sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            pooled = attended_out.mean(dim=1)
            
        # Классификация
        output = self.classifier(pooled)
        
        return output, attention_weights

class GestureDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, class_to_idx=None):
        self.sequences = sequences
        self.labels = labels
        
        if class_to_idx is None:
            unique_labels = list(set(labels))
            self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.class_to_idx = class_to_idx
            
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}
        
        # Преобразуем метки в индексы
        self.label_indices = [self.class_to_idx[label] for label in labels]
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.label_indices[idx]])
        length = torch.LongTensor([len(self.sequences[idx])])
        
        return sequence, label.squeeze(), length.squeeze()

def collate_fn(batch):
    """
    Функция для объединения образцов в батч с паддингом
    """
    sequences, labels, lengths = zip(*batch)
    
    # Паддинг последовательностей
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    # Сортировка по длине для эффективной обработки
    lengths = torch.stack(lengths)
    labels = torch.stack(labels)
    
    # Сортируем по убыванию длины
    sorted_lengths, sorted_idx = lengths.sort(0, descending=True)
    sorted_sequences = padded_sequences[sorted_idx]
    sorted_labels = labels[sorted_idx]
    
    return sorted_sequences, sorted_labels, sorted_lengths

class GesturePreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, sequences):
        """Вычисляет статистики для нормализации"""
        all_data = np.concatenate([seq for seq in sequences], axis=0)
        self.mean = np.mean(all_data, axis=0)
        self.std = np.std(all_data, axis=0) + 1e-8  # избегаем деления на ноль
        
    def transform(self, sequences):
        """Применяет нормализацию к последовательностям"""
        if self.mean is None or self.std is None:
            raise ValueError("Preprocessor не обучен. Вызовите fit() сначала.")
            
        normalized_sequences = []
        for seq in sequences:
            normalized_seq = (seq - self.mean) / self.std
            normalized_sequences.append(normalized_seq)
            
        return normalized_sequences
    
    def fit_transform(self, sequences):
        """Обучает и применяет нормализацию"""
        self.fit(sequences)
        return self.transform(sequences)

def augment_sequence(sequence, noise_factor=0.01, time_stretch_factor=0.1):
    """
    Аугментация последовательности жестов
    """
    augmented_sequences = [sequence]  # Оригинальная последовательность
    
    # Добавление шума
    noise = np.random.normal(0, noise_factor, sequence.shape)
    noisy_sequence = sequence + noise
    augmented_sequences.append(noisy_sequence)
    
    # Временное растяжение/сжатие
    seq_len = len(sequence)
    new_len = int(seq_len * (1 + np.random.uniform(-time_stretch_factor, time_stretch_factor)))
    new_len = max(5, min(new_len, seq_len * 2))  # Ограничиваем изменения
    
    indices = np.linspace(0, seq_len - 1, new_len)
    stretched_sequence = np.array([sequence[int(i)] for i in indices])
    augmented_sequences.append(stretched_sequence)
    
    return augmented_sequences

def create_model(num_classes=2, input_size=84):
    """Создает модель с оптимальными параметрами"""
    model = GestureLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3
    )
    return model

if __name__ == "__main__":
    # Тестирование модели
    model = create_model()
    
    # Тестовые данные
    batch_size = 4
    seq_len = 30
    input_size = 84
    
    x = torch.randn(batch_size, seq_len, input_size)
    lengths = torch.randint(10, seq_len, (batch_size,))
    
    output, attention = model(x, lengths)
    print(f"Выходной тензор: {output.shape}")
    print(f"Attention веса: {attention.shape}")
    print("Модель создана успешно!")