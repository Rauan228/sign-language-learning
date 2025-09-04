# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
from gesture_lstm_model import GestureLSTM

class AllClassesGestureDataset(Dataset):
    def __init__(self, sequences, labels, scaler=None):
        self.sequences = sequences
        self.labels = labels
        self.scaler = scaler
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Нормализация если есть scaler
        if self.scaler is not None:
            original_shape = sequence.shape
            sequence_flat = sequence.reshape(-1, sequence.shape[-1])
            sequence_normalized = self.scaler.transform(sequence_flat)
            sequence = sequence_normalized.reshape(original_shape)
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])

def extract_hand_keypoints(video_path):
    """Извлечение ключевых точек рук из видео"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        frame_keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    frame_keypoints.extend([landmark.x, landmark.y, landmark.z])
        
        # Нормализуем количество ключевых точек до фиксированного размера
        if len(frame_keypoints) == 0:
            # Если руки не обнаружены, заполняем нулями
            frame_keypoints = [0.0] * 126  # 21 точка * 3 координаты * 2 руки
        elif len(frame_keypoints) == 63:
            # Одна рука обнаружена, добавляем нули для второй руки
            frame_keypoints.extend([0.0] * 63)
        elif len(frame_keypoints) > 126:
            # Если обнаружено больше точек, обрезаем до 126
            frame_keypoints = frame_keypoints[:126]
        elif len(frame_keypoints) < 126 and len(frame_keypoints) > 63:
            # Если между 63 и 126, дополняем до 126
            frame_keypoints.extend([0.0] * (126 - len(frame_keypoints)))
        
        # Убеждаемся, что у нас ровно 126 элементов
        frame_keypoints = frame_keypoints[:126] + [0.0] * max(0, 126 - len(frame_keypoints))
        
        keypoints_sequence.append(frame_keypoints)
    
    cap.release()
    hands.close()
    
    # Преобразуем в numpy array с проверкой формы
    if len(keypoints_sequence) > 0:
        return np.array(keypoints_sequence, dtype=np.float32)
    else:
        return np.array([], dtype=np.float32).reshape(0, 126)

def load_no_event_data(test_dir, annotations_file, max_samples=50):
    """Загрузка данных no_event из папки test"""
    print("=== Загрузка данных no_event ===")
    
    sequences = []
    labels = []
    
    # Читаем аннотации
    try:
        df = pd.read_csv(annotations_file, sep='\t', on_bad_lines='skip')
    except:
        try:
            df = pd.read_csv(annotations_file, sep=',', on_bad_lines='skip')
        except Exception as e:
            print(f"Ошибка чтения аннотаций: {e}")
            return [], []
    
    # Фильтруем no_event записи
    no_event_records = df[df['text'] == 'no_event'].head(max_samples)
    print(f"Найдено {len(no_event_records)} записей no_event")
    
    processed_count = 0
    for _, row in no_event_records.iterrows():
        attachment_id = str(row['attachment_id'])
        
        # Убираем префикс 'no' если есть
        if attachment_id.startswith('no'):
            file_id = attachment_id[2:]
        else:
            file_id = attachment_id
            
        video_path = os.path.join(test_dir, f"{file_id}.mp4")
        
        if os.path.exists(video_path):
            print(f"Обрабатываю no_event: {video_path}")
            keypoints = extract_hand_keypoints(video_path)
            
            if len(keypoints) > 0:
                # Создаем последовательности фиксированной длины
                sequence_length = 30
                
                if len(keypoints) >= sequence_length:
                    # Если видео длиннее, берем несколько перекрывающихся окон
                    step = max(1, len(keypoints) // 3)  # 3 окна на видео
                    for start_idx in range(0, len(keypoints) - sequence_length + 1, step):
                        sequence = keypoints[start_idx:start_idx + sequence_length]
                        sequences.append(sequence)
                        labels.append(6)  # no_event будет класс 6
                else:
                    # Если видео короче, дополняем повторением последнего кадра
                    padded_sequence = np.zeros((sequence_length, keypoints.shape[1]))
                    padded_sequence[:len(keypoints)] = keypoints
                    if len(keypoints) > 0:
                        padded_sequence[len(keypoints):] = keypoints[-1]
                    sequences.append(padded_sequence)
                    labels.append(6)  # no_event будет класс 6
                
                processed_count += 1
                print(f"Обработано no_event {video_path}: {len(keypoints)} кадров")
        else:
            print(f"Файл не найден: {video_path}")
    
    print(f"Обработано {processed_count} файлов no_event")
    return sequences, labels

def load_gesture_data(train_dir):
    """Загрузка данных жестов из папок"""
    sequences = []
    labels = []
    # Все классы: 6 жестов + no_event
    class_names = ['выходные', 'кусок', 'любить', 'наконец-то', 'неожиданный', 'осень']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    # no_event будет иметь индекс 6
    class_to_idx['no_event'] = 6
    class_names.append('no_event')
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}
    
    print("=== Загрузка данных жестов ===")
    
    for class_name in class_names[:-1]:  # Исключаем no_event, его загружаем отдельно
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Папка {class_dir} не найдена!")
            continue
            
        print(f"Обрабатываю класс '{class_name}'...")
        
        for video_file in os.listdir(class_dir):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(class_dir, video_file)
                print(f"Обрабатываю {video_path}...")
                
                keypoints = extract_hand_keypoints(video_path)
                if len(keypoints) > 0:
                    # Создаем последовательности фиксированной длины
                    sequence_length = 30
                    
                    if len(keypoints) >= sequence_length:
                        # Если видео длиннее, берем несколько перекрывающихся окон
                        step = max(1, len(keypoints) // 5)  # 5 окон на видео
                        for start_idx in range(0, len(keypoints) - sequence_length + 1, step):
                            sequence = keypoints[start_idx:start_idx + sequence_length]
                            sequences.append(sequence)
                            labels.append(class_to_idx[class_name])
                    else:
                        # Если видео короче, дополняем повторением последнего кадра
                        padded_sequence = np.zeros((sequence_length, keypoints.shape[1]))
                        padded_sequence[:len(keypoints)] = keypoints
                        if len(keypoints) > 0:
                            padded_sequence[len(keypoints):] = keypoints[-1]
                        sequences.append(padded_sequence)
                        labels.append(class_to_idx[class_name])
                        
                    print(f"Обработано {video_path}: {len(keypoints)} кадров")
    
    return sequences, labels, class_to_idx, idx_to_class

def augment_data(sequences, labels, target_samples_per_class=20):
    """Аугментация данных для балансировки классов"""
    print("\n=== Аугментация данных ===")
    
    augmented_sequences = []
    augmented_labels = []
    
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        class_sequences = [seq for seq, lbl in zip(sequences, labels) if lbl == label]
        class_labels = [lbl for lbl in labels if lbl == label]
        
        # Добавляем оригинальные данные
        augmented_sequences.extend(class_sequences)
        augmented_labels.extend(class_labels)
        
        # Если нужно больше данных для этого класса
        current_count = len(class_sequences)
        if current_count < target_samples_per_class:
            needed = target_samples_per_class - current_count
            
            for _ in range(needed):
                # Выбираем случайную последовательность из класса
                idx = np.random.randint(0, len(class_sequences))
                original_seq = class_sequences[idx].copy()
                
                # Применяем аугментацию
                augmented_seq = original_seq.copy()
                
                # Добавляем небольшой шум
                noise = np.random.normal(0, 0.01, augmented_seq.shape)
                augmented_seq += noise
                
                # Небольшое масштабирование
                scale_factor = np.random.uniform(0.95, 1.05)
                augmented_seq *= scale_factor
                
                augmented_sequences.append(augmented_seq)
                augmented_labels.append(label)
    
    augmented_sequences = np.array(augmented_sequences)
    augmented_labels = np.array(augmented_labels)
    
    print("После аугментации:")
    print(f"Всего последовательностей: {len(augmented_sequences)}")
    for label in unique_labels:
        count = sum(1 for l in augmented_labels if l == label)
        print(f"Класс {label}: {count}")
    
    return augmented_sequences, augmented_labels

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """Обучение модели"""
    print("\n=== Начало обучения ===")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    early_stop_threshold = 0.95  # Останавливаем при достижении 95% точности
    
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_bar):
            data, target = data.to(device), target.squeeze().to(device)
            
            optimizer.zero_grad()
            output, _ = model(data)  # Модель возвращает (output, attention_weights)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for data, target in val_bar:
                data, target = data.to(device), target.squeeze().to(device)
                output, _ = model(data)  # Модель возвращает (output, attention_weights)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Сохраняем метрики
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Обновляем learning rate
        scheduler.step(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, 'best_all_classes_model.pth')
            print(f'Сохранена новая лучшая модель с точностью: {val_acc:.2f}%')
        
        # Ранняя остановка при достижении 95% точности
        if val_acc >= early_stop_threshold * 100:
            print(f'\nДостигнута точность {val_acc:.2f}% (>= {early_stop_threshold*100:.0f}%). Останавливаем обучение.')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, 'best_all_classes_model.pth')
            print(f'Финальная модель сохранена с точностью: {val_acc:.2f}%')
            break
    
    return train_losses, train_accuracies, val_losses, val_accuracies, best_val_acc

def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies, save_path):
    """Построение графиков обучения"""
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    
    # График потерь
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Потери во время обучения')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Потери')
    ax1.legend()
    ax1.grid(True)
    
    # График точности
    ax2.plot(train_accuracies, label='Train Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title('Точность во время обучения')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"График сохранен: {save_path}")

def main():
    """Основная функция"""
    print("Скрипт обучения модели для всех классов жестов + no_event")
    print("Классы: выходные, кусок, любить, наконец-то, неожиданный, осень, no_event")
    
    # Настройки
    train_dir = 'D:/gesture/train'
    test_dir = 'D:/gesture/test'
    annotations_file = 'D:/gesture/annotations.csv'
    output_dir = 'D:/gesture/all_classes_model_output'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Создаем папку для выходных файлов
    os.makedirs(output_dir, exist_ok=True)
    
    # Загрузка данных жестов
    gesture_sequences, gesture_labels, class_to_idx, idx_to_class = load_gesture_data(train_dir)
    
    # Загрузка данных no_event
    no_event_sequences, no_event_labels = load_no_event_data(test_dir, annotations_file, max_samples=50)
    
    # Объединяем все данные
    all_sequences = gesture_sequences + no_event_sequences
    all_labels = gesture_labels + no_event_labels
    
    if len(all_sequences) == 0:
        print("Ошибка: Не удалось загрузить данные!")
        return
    
    print(f"\nВсего последовательностей: {len(all_sequences)}")
    for class_name, class_idx in class_to_idx.items():
        count = sum(1 for label in all_labels if label == class_idx)
        print(f"Класс '{class_name}': {count}")
    
    # Преобразуем в numpy arrays
    sequences = np.array(all_sequences)
    labels = np.array(all_labels)
    
    print(f"\nФорма данных: {sequences.shape}")
    print(f"Форма меток: {labels.shape}")
    
    # Аугментация данных
    sequences, labels = augment_data(sequences, labels, target_samples_per_class=30)
    
    # Нормализация данных
    print("\n=== Нормализация данных ===")
    original_shape = sequences.shape
    sequences_flat = sequences.reshape(-1, sequences.shape[-1])
    
    scaler = StandardScaler()
    sequences_normalized = scaler.fit_transform(sequences_flat)
    sequences = sequences_normalized.reshape(original_shape)
    
    # Разделение на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nОбучающая выборка: {X_train.shape}")
    print(f"Валидационная выборка: {X_val.shape}")
    
    # Создание датасетов
    train_dataset = AllClassesGestureDataset(X_train, y_train)
    val_dataset = AllClassesGestureDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Создание модели
    print("\n=== Создание модели ===")
    input_size = sequences.shape[-1]  # 126 (21 точка * 3 координаты * 2 руки)
    hidden_size = 128
    num_layers = 2
    num_classes = len(class_to_idx)  # 7 классов
    
    model = GestureLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    print(f"Модель создана с {num_classes} классами")
    
    # Обучение модели
    train_losses, train_accuracies, val_losses, val_accuracies, best_val_acc = train_model(
        model, train_loader, val_loader, num_epochs=50, device=device
    )
    
    # Загрузка лучшей модели
    checkpoint = torch.load('best_all_classes_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Сохранение модели и конфигурации
    print("\n=== Сохранение модели ===")
    
    # Сохранение финальной модели
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_classes': num_classes,
        'best_accuracy': best_val_acc
    }, os.path.join(output_dir, 'all_classes_gesture_model.pth'))
    
    # Сохранение конфигурации
    config = {
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_classes': num_classes,
        'best_accuracy': best_val_acc,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # Построение графиков
    plot_training_history(
        train_losses, train_accuracies, val_losses, val_accuracies,
        os.path.join(output_dir, 'training_history.png')
    )
    
    print(f"\n🎉 Обучение завершено!")
    print(f"Лучшая точность: {best_val_acc:.2f}%")
    print(f"Модель сохранена в: {output_dir}")
    print(f"Классы: {list(class_to_idx.keys())}")

if __name__ == '__main__':
    main()