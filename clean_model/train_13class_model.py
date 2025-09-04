# -*- coding: utf-8 -*-
"""
Скрипт обучения модели LSTM для распознавания 13 классов жестов
Адаптирован для структуры папок с жестами
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe as mp
import json
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Импорт модели
from gesture_lstm_model_13class import GestureLSTM

class GestureDataset(Dataset):
    """Dataset для последовательностей жестов"""
    def __init__(self, sequences, labels, transform=None):
        self.sequences = sequences
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])

def extract_hand_keypoints(video_path, max_frames=30):
    """
    Извлечение ключевых точек рук из видео
    Возвращает последовательность из 126 признаков (2 руки * 21 точка * 3 координаты)
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    
    frame_count = 0
    while cap.read()[0] and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Конвертация BGR в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Инициализация массива для ключевых точек (126 признаков)
        frame_keypoints = np.zeros(126)
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= 2:  # Максимум 2 руки
                    break
                
                # Извлечение координат для каждой руки
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    hand_points.extend([landmark.x, landmark.y, landmark.z])
                
                # Размещение в соответствующей позиции (63 признака на руку)
                start_idx = hand_idx * 63
                frame_keypoints[start_idx:start_idx + 63] = hand_points
        
        keypoints_sequence.append(frame_keypoints)
        frame_count += 1
    
    cap.release()
    hands.close()
    
    # Дополнение или обрезка до max_frames
    if len(keypoints_sequence) < max_frames:
        # Дополнение нулями
        while len(keypoints_sequence) < max_frames:
            keypoints_sequence.append(np.zeros(126))
    else:
        # Обрезка до max_frames
        keypoints_sequence = keypoints_sequence[:max_frames]
    
    return np.array(keypoints_sequence)

def load_gesture_data(train_dir, max_frames=30):
    """
    Загрузка данных жестов из папок
    """
    sequences = []
    labels = []
    
    # Получение списка папок с жестами
    gesture_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
    print(f"Найдено папок с жестами: {len(gesture_folders)}")
    print(f"Жесты: {gesture_folders}")
    
    for gesture_name in gesture_folders:
        gesture_path = os.path.join(train_dir, gesture_name)
        video_files = [f for f in os.listdir(gesture_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"Обработка жеста '{gesture_name}': {len(video_files)} видео")
        
        for video_file in tqdm(video_files, desc=f"Жест {gesture_name}"):
            video_path = os.path.join(gesture_path, video_file)
            try:
                keypoints = extract_hand_keypoints(video_path, max_frames)
                sequences.append(keypoints)
                labels.append(gesture_name)
            except Exception as e:
                print(f"Ошибка обработки {video_file}: {e}")
                continue
    
    return sequences, labels

def load_no_event_data(annotations_file, test_dir, num_samples=200, max_frames=30):
    """
    Загрузка данных для класса 'no_event' из аннотированных видео
    """
    import pandas as pd
    
    no_event_sequences = []
    no_event_labels = []
    
    if not os.path.exists(annotations_file):
        print(f"Файл аннотаций не найден: {annotations_file}")
        return no_event_sequences, no_event_labels
    
    # Читаем аннотации
    annotations = pd.read_csv(annotations_file, sep='\t')
    
    # Фильтруем только no_event записи
    no_event_annotations = annotations[annotations['text'] == 'no_event']
    print(f"Найдено no_event аннотаций: {len(no_event_annotations)}")
    
    # Ограничиваем количество для ускорения обучения
    if len(no_event_annotations) > num_samples:
        no_event_annotations = no_event_annotations.sample(n=num_samples, random_state=42)
        print(f"Используем {num_samples} случайных no_event аннотаций")
    
    for _, row in tqdm(no_event_annotations.iterrows(), total=len(no_event_annotations), desc="no_event видео"):
        attachment_id = row['attachment_id']
        
        # Пробуем найти видео файл с разными расширениями
        video_path = None
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            potential_path = os.path.join(test_dir, f"{attachment_id}{ext}")
            if os.path.exists(potential_path):
                video_path = potential_path
                break
        
        if video_path is None:
            continue
            
        try:
            keypoints = extract_hand_keypoints(video_path, max_frames)
            no_event_sequences.append(keypoints)
            no_event_labels.append('no_event')
        except Exception as e:
            continue
    
    print(f"Успешно обработано {len(no_event_sequences)} no_event видео")
    return no_event_sequences, no_event_labels

def augment_sequence(sequence, noise_factor=0.01, time_stretch_factor=0.1):
    """
    Аугментация последовательности
    """
    augmented = sequence.copy()
    
    # Добавление шума
    noise = np.random.normal(0, noise_factor, sequence.shape)
    augmented += noise
    
    # Временное растяжение/сжатие
    if np.random.random() > 0.5:
        stretch_factor = 1 + np.random.uniform(-time_stretch_factor, time_stretch_factor)
        new_length = int(len(sequence) * stretch_factor)
        if new_length > 0:
            indices = np.linspace(0, len(sequence) - 1, new_length).astype(int)
            augmented = sequence[indices]
            
            # Приведение к исходной длине
            if len(augmented) < len(sequence):
                # Дополнение
                padding = np.tile(augmented[-1], (len(sequence) - len(augmented), 1))
                augmented = np.vstack([augmented, padding])
            else:
                # Обрезка
                augmented = augmented[:len(sequence)]
    
    return augmented

def main():
    # Параметры
    TRAIN_DIR = "D:/gesture/train"
    TEST_DIR = "D:/gesture/test"
    ANNOTATIONS_FILE = "D:/gesture/annotations.csv"  # Файл с аннотациями
    OUTPUT_DIR = "D:/gesture/13class_model_output"
    
    MAX_FRAMES = 30
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Создание выходной директории
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Загрузка данных жестов...")
    
    # Загрузка данных жестов
    sequences, labels = load_gesture_data(TRAIN_DIR, MAX_FRAMES)
    
    # Загрузка данных no_event из аннотированных видео
    print("\nЗагрузка данных no_event из аннотированных видео...")
    no_event_sequences, no_event_labels = load_no_event_data(ANNOTATIONS_FILE, TEST_DIR, num_samples=200, max_frames=MAX_FRAMES)
    
    sequences.extend(no_event_sequences)
    labels.extend(no_event_labels)
    
    print(f"\nВсего последовательностей: {len(sequences)}")
    
    if len(sequences) == 0:
        print("Ошибка: не найдено данных для обучения!")
        return
    
    # Создание маппинга классов
    unique_labels = sorted(list(set(labels)))
    class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    
    print(f"Классы ({len(unique_labels)}): {unique_labels}")
    
    # Подсчет количества образцов по классам
    from collections import Counter
    label_counts = Counter(labels)
    print("\nРаспределение классов:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # Преобразование меток в индексы
    label_indices = [class_to_idx[label] for label in labels]
    
    # Преобразование в numpy массивы
    X = np.array(sequences)
    y = np.array(label_indices)
    
    print(f"\nФорма данных: {X.shape}")
    print(f"Форма меток: {y.shape}")
    
    # Нормализация данных
    print("Нормализация данных...")
    scaler = StandardScaler()
    
    # Reshape для нормализации
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_normalized = scaler.fit_transform(X_reshaped)
    X_normalized = X_normalized.reshape(X.shape)
    
    # Разделение на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Обучающая выборка: {X_train.shape}")
    print(f"Валидационная выборка: {X_val.shape}")
    
    # Аугментация данных
    print("Аугментация данных...")
    X_train_aug = []
    y_train_aug = []
    
    for i in range(len(X_train)):
        # Оригинальная последовательность
        X_train_aug.append(X_train[i])
        y_train_aug.append(y_train[i])
        
        # Аугментированная последовательность
        aug_seq = augment_sequence(X_train[i])
        X_train_aug.append(aug_seq)
        y_train_aug.append(y_train[i])
    
    X_train_aug = np.array(X_train_aug)
    y_train_aug = np.array(y_train_aug)
    
    print(f"После аугментации: {X_train_aug.shape}")
    
    # Создание DataLoader'ов
    train_dataset = GestureDataset(X_train_aug, y_train_aug)
    val_dataset = GestureDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Создание модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    model = GestureLSTM(
        input_size=126,
        hidden_size=128,
        num_layers=2,
        num_classes=len(unique_labels),
        dropout=0.3
    ).to(device)
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Обучение
    print("\nНачало обучения...")
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_accuracy = 0.0
    
    for epoch in range(EPOCHS):
        # Обучение
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.squeeze().to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.squeeze().to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Эпоха {epoch+1}/{EPOCHS}:')
        print(f'  Потери обучения: {train_loss:.4f}')
        print(f'  Потери валидации: {val_loss:.4f}')
        print(f'  Точность валидации: {val_accuracy:.2f}%')
        
        # Сохранение лучшей модели
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': val_accuracy
            }, os.path.join(OUTPUT_DIR, '13class_gesture_model.pth'))
            print(f'  Новая лучшая модель сохранена! Точность: {val_accuracy:.2f}%')
        
        scheduler.step(val_accuracy)
        print()
    
    print(f"Обучение завершено! Лучшая точность: {best_accuracy:.2f}%")
    
    # Сохранение конфигурации
    config = {
        'input_size': 126,
        'hidden_size': 128,
        'num_layers': 2,
        'num_classes': len(unique_labels),
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'best_accuracy': best_accuracy,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }
    
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # Сохранение scaler
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Построение графиков
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Обучение')
    plt.plot(val_losses, label='Валидация')
    plt.title('Потери')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies)
    plt.title('Точность валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (%)')
    
    # Финальная оценка
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.squeeze().to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Матрица ошибок
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[idx_to_class[i] for i in range(len(unique_labels))],
                yticklabels=[idx_to_class[i] for i in range(len(unique_labels))])
    plt.title('Матрица ошибок')
    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Отчет по классификации
    print("\nОтчет по классификации:")
    print(classification_report(all_targets, all_preds, 
                              target_names=[idx_to_class[i] for i in range(len(unique_labels))]))
    
    print(f"\nВсе файлы сохранены в: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()