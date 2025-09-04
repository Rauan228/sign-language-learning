# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import json
import os
from collections import deque
from sklearn.preprocessing import StandardScaler
from clean_model.gesture_lstm_model import GestureLSTM
import time
import threading
from queue import Queue

# Функция для отображения русского текста
def put_russian_text(img, text, position, font_scale=1, color=(255, 255, 255), thickness=2):
    """Отображение русского текста на изображении"""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Конвертируем OpenCV изображение в PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        # Пытаемся загрузить шрифт, поддерживающий кириллицу
        font = ImageFont.truetype("arial.ttf", int(30 * font_scale))
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", int(30 * font_scale))
        except:
            font = ImageFont.load_default()
    
    # Рисуем текст
    draw.text(position, text, font=font, fill=color[::-1])  # BGR -> RGB
    
    # Конвертируем обратно в OpenCV формат
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv

class ThirteenClassGestureRecognizer:
    def __init__(self, model_path, config_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        
        # Загрузка конфигурации
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.class_to_idx = self.config['class_to_idx']
        self.idx_to_class = {int(k): v for k, v in self.config['idx_to_class'].items()}
        
        # Создание и загрузка модели
        self.model = GestureLSTM(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            num_classes=self.config['num_classes']
        ).to(self.device)
        
        # Загрузка весов модели
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Создание scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(self.config['scaler_mean'])
        self.scaler.scale_ = np.array(self.config['scaler_scale'])
        
        # Инициализация MediaPipe с оптимизированными настройками
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,  # Снижено для лучшего трекинга
            min_tracking_confidence=0.3,   # Снижено для стабильности
            model_complexity=0              # Быстрая модель
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Параметры оптимизации
        self.frame_skip = 2  # Обрабатывать каждый 2-й кадр
        self.frame_counter = 0
        self.processing_frame = None
        self.last_keypoints = None
        self.stable_hands_count = 0
        self.min_stable_frames = 3  # Минимум кадров для стабильного трекинга
        
        # Буфер для последовательности кадров
        self.sequence_length = 30
        self.keypoints_buffer = deque(maxlen=self.sequence_length)
        
        # Параметры для фильтрации предсказаний
        self.prediction_buffer = deque(maxlen=10)
        self.confidence_threshold = {
            'no_event': 0.6,  # Более низкий порог для no_event
            'gesture': 0.7    # Более высокий порог для жестов
        }
        
        # Состояние распознавания
        self.current_gesture = "Ожидание..."
        self.current_confidence = 0.0
        self.last_prediction_time = time.time()
        self.prediction_cooldown = 0.3  # Оптимизировано для быстрого отклика
        
        # Кэш для текста (оптимизация отрисовки)
        self.text_cache = {}
        self.last_text_update = 0
        self.text_update_interval = 0.05  # Обновлять текст каждые 50мс для лучшей отзывчивости
        
        # Дополнительные оптимизации
        self.gesture_stability_counter = 0
        self.min_gesture_stability = 3
        self.last_stable_gesture = "Ожидание..."
        
        print(f"Модель загружена. Классы: {list(self.class_to_idx.keys())}")
        print(f"Лучшая точность модели: {self.config.get('best_accuracy', 'N/A'):.2f}%")
    
    def extract_keypoints(self, frame):
        """Извлечение ключевых точек из кадра с оптимизацией"""
        # Уменьшаем размер кадра для ускорения обработки
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_small = cv2.resize(frame, (new_width, new_height))
        else:
            frame_small = frame
            
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        keypoints = []
        hands_detected = 0
        
        if results.multi_hand_landmarks:
            hands_detected = len(results.multi_hand_landmarks)
            
            # Сортируем руки по x-координате для консистентности
            hand_data = []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_keypoints = []
                avg_x = sum(landmark.x for landmark in hand_landmarks.landmark) / len(hand_landmarks.landmark)
                for landmark in hand_landmarks.landmark:
                    hand_keypoints.extend([landmark.x, landmark.y, landmark.z])
                hand_data.append((avg_x, hand_keypoints))
            
            # Сортируем по x-координате (левая рука первая)
            hand_data.sort(key=lambda x: x[0])
            
            for _, hand_keypoints in hand_data:
                keypoints.extend(hand_keypoints)
        
        # Стабилизация трекинга
        if hands_detected > 0:
            self.stable_hands_count = min(self.stable_hands_count + 1, self.min_stable_frames)
        else:
            self.stable_hands_count = max(self.stable_hands_count - 1, 0)
            
        # Используем последние стабильные keypoints если трекинг нестабилен
        if self.stable_hands_count < self.min_stable_frames and self.last_keypoints is not None:
            keypoints = self.last_keypoints.copy()
        elif len(keypoints) > 0:
            self.last_keypoints = keypoints.copy()
        
        # Нормализуем до фиксированного размера (126 элементов для 2 рук)
        if len(keypoints) == 0:
            keypoints = [0.0] * 126
        elif len(keypoints) == 63:  # Одна рука
            keypoints.extend([0.0] * 63)
        elif len(keypoints) > 126:  # Больше 2 рук (обрезаем)
            keypoints = keypoints[:126]
        elif len(keypoints) < 126 and len(keypoints) > 63:
            keypoints.extend([0.0] * (126 - len(keypoints)))
        
        # Убеждаемся, что у нас ровно 126 элементов
        keypoints = keypoints[:126] + [0.0] * max(0, 126 - len(keypoints))
        
        return np.array(keypoints, dtype=np.float32), results, hands_detected
    
    def predict_gesture(self, sequence):
        """Предсказание жеста по последовательности с оптимизацией"""
        if len(sequence) < self.sequence_length:
            return None, 0.0
        
        try:
            # Подготовка данных (оптимизированная)
            sequence_array = np.array(sequence, dtype=np.float32)
            
            # Нормализация
            original_shape = sequence_array.shape
            sequence_flat = sequence_array.reshape(-1, sequence_array.shape[-1])
            sequence_normalized = self.scaler.transform(sequence_flat)
            sequence_normalized = sequence_normalized.reshape(original_shape)
            
            # Предсказание с оптимизацией памяти
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0)
                if self.device.type == 'cuda':
                    sequence_tensor = sequence_tensor.to(self.device, non_blocking=True)
                else:
                    sequence_tensor = sequence_tensor.to(self.device)
                
                outputs, _ = self.model(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = predicted.item()
                confidence_value = confidence.item()
                
                # Очистка GPU памяти
                if self.device.type == 'cuda':
                    del sequence_tensor, outputs, probabilities
                    torch.cuda.empty_cache()
                
                return predicted_class, confidence_value
                
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return None, 0.0
    
    def filter_predictions(self, predicted_class, confidence):
        """Улучшенная фильтрация предсказаний для стабильности"""
        if predicted_class is None:
            return self.current_gesture, self.current_confidence
        
        gesture_name = self.idx_to_class[predicted_class]
        
        # Применяем разные пороги уверенности
        if gesture_name == 'no_event':
            threshold = self.confidence_threshold['no_event']
        else:
            threshold = self.confidence_threshold['gesture']
        
        if confidence < threshold:
            # Сбрасываем счетчик стабильности при низкой уверенности
            self.gesture_stability_counter = 0
            return self.current_gesture, self.current_confidence
        
        # Проверяем стабильность жеста
        if gesture_name == self.last_stable_gesture:
            self.gesture_stability_counter += 1
        else:
            self.gesture_stability_counter = 1
            self.last_stable_gesture = gesture_name
        
        # Добавляем в буфер предсказаний
        self.prediction_buffer.append((gesture_name, confidence))
        
        # Требуем минимальную стабильность для смены жеста
        if self.gesture_stability_counter >= self.min_gesture_stability:
            # Анализируем последние предсказания для дополнительной стабильности
            if len(self.prediction_buffer) >= 3:
                recent_predictions = list(self.prediction_buffer)[-3:]
                gesture_counts = {}
                
                for gesture, conf in recent_predictions:
                    if gesture not in gesture_counts:
                        gesture_counts[gesture] = {'count': 0, 'confidence': 0}
                    gesture_counts[gesture]['count'] += 1
                    gesture_counts[gesture]['confidence'] += conf
                
                # Находим наиболее частый жест в последних предсказаниях
                if gesture_counts:
                    most_frequent = max(gesture_counts.items(), key=lambda x: x[1]['count'])
                    if most_frequent[1]['count'] >= 2:  # Появляется в большинстве
                        avg_confidence = most_frequent[1]['confidence'] / most_frequent[1]['count']
                        return most_frequent[0], avg_confidence
            
            return gesture_name, confidence
        
        # Возвращаем текущий жест если новый недостаточно стабилен
        return self.current_gesture, self.current_confidence
    
    def run(self):
        """Запуск распознавания в реальном времени с оптимизацией"""
        cap = cv2.VideoCapture(0)
        # Уменьшаем разрешение для лучшей производительности
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Минимальный буфер
        
        print("Запуск оптимизированного распознавания жестов (13 классов)...")
        print("Классы: веселиться, воскресенье, вторник, выходные, любить, осень,")
        print("        понедельник, пятница, редко, среда, суббота, четверг, no_event")
        print("Нажмите 'q' для выхода")
        print("Нажмите 'r' для сброса буфера")
        print("Нажмите 's' для отображения статистики")
        
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Отражаем кадр горизонтально для удобства
            frame = cv2.flip(frame, 1)
            
            # Пропускаем кадры для оптимизации
            self.frame_counter += 1
            process_frame = (self.frame_counter % self.frame_skip == 0)
            
            if process_frame:
                # Извлекаем ключевые точки
                keypoints, results, hands_detected = self.extract_keypoints(frame)
                
                # Добавляем в буфер только если есть стабильный трекинг
                if self.stable_hands_count >= self.min_stable_frames or hands_detected > 0:
                    self.keypoints_buffer.append(keypoints)
                
                # Сохраняем результаты для отрисовки
                self.processing_frame = (results, hands_detected)
            
            # Рисуем ключевые точки рук (используем последние результаты)
            if self.processing_frame and self.processing_frame[0].multi_hand_landmarks:
                for hand_landmarks in self.processing_frame[0].multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                            color=(255, 255, 255), thickness=2))
            
            # Предсказание жеста (только при обработке кадра)
            current_time = time.time()
            if (process_frame and len(self.keypoints_buffer) == self.sequence_length and 
                current_time - self.last_prediction_time > self.prediction_cooldown):
                
                predicted_class, confidence = self.predict_gesture(list(self.keypoints_buffer))
                filtered_gesture, filtered_confidence = self.filter_predictions(predicted_class, confidence)
                
                self.current_gesture = filtered_gesture
                self.current_confidence = filtered_confidence
                self.last_prediction_time = current_time
            
            # Подсчет FPS
            fps_counter += 1
            if current_time - fps_start_time >= 1.0:
                current_fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            
            # Отображение информации (оптимизированное)
            info_y = 30
            
            # Обновляем текстовый кэш только периодически
            if current_time - self.last_text_update > self.text_update_interval:
                self.text_cache = {
                    'title': "Распознавание жестов (13 классов) - Оптимизировано",
                    'gesture': f"Жест: {self.current_gesture}",
                    'confidence': f"Уверенность: {self.current_confidence:.2f}",
                    'buffer': f"Буфер: {len(self.keypoints_buffer)}/{self.sequence_length}",
                    'hands': f"Руки: {self.processing_frame[1] if self.processing_frame else 0}",
                    'fps': f"FPS: {current_fps:.1f}",
                    'stability': f"Трекинг: {self.stable_hands_count}/{self.min_stable_frames}",
                    'gesture_stability': f"Стабильность жеста: {self.gesture_stability_counter}/{self.min_gesture_stability}"
                }
                self.last_text_update = current_time
            
            # Заголовок
            cv2.putText(frame, self.text_cache['title'], (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 30
            
            # Текущий жест
            gesture_color = (0, 255, 0) if self.current_confidence > 0.7 else (0, 165, 255)
            frame = put_russian_text(frame, self.text_cache['gesture'], (10, info_y), 
                                   font_scale=0.7, color=gesture_color, thickness=2)
            info_y += 35
            
            # Уверенность
            cv2.putText(frame, self.text_cache['confidence'], (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, gesture_color, 2)
            info_y += 25
            
            # Статус буфера и рук
            cv2.putText(frame, self.text_cache['buffer'], (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            info_y += 20
            
            cv2.putText(frame, self.text_cache['hands'], (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            info_y += 20
            
            # FPS и стабильность
            cv2.putText(frame, self.text_cache['fps'], (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            info_y += 20
            
            cv2.putText(frame, self.text_cache['stability'], (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            info_y += 20
            
            cv2.putText(frame, self.text_cache['gesture_stability'], (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            info_y += 25
            
            # Инструкции (компактно)
            cv2.putText(frame, "'q'-выход 'r'-сброс 's'-статистика", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Индикаторы состояния (компактные)
            # Статус буфера
            buffer_color = (0, 255, 0) if len(self.keypoints_buffer) == self.sequence_length else (0, 0, 255)
            cv2.circle(frame, (frame.shape[1] - 60, 20), 8, buffer_color, -1)
            
            # Статус трекинга
            tracking_color = (0, 255, 0) if self.stable_hands_count >= self.min_stable_frames else (255, 0, 0)
            cv2.circle(frame, (frame.shape[1] - 30, 20), 8, tracking_color, -1)
            
            cv2.imshow('Gesture Recognition - 13 Classes', frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.keypoints_buffer.clear()
                self.prediction_buffer.clear()
                self.current_gesture = "Ожидание..."
                self.current_confidence = 0.0
                self.gesture_stability_counter = 0
                self.last_stable_gesture = "Ожидание..."
                self.stable_hands_count = 0
                print("Все буферы и счетчики сброшены")
            elif key == ord('s'):
                print(f"\n=== СТАТИСТИКА ===")
                print(f"Текущий жест: {self.current_gesture}")
                print(f"Уверенность: {self.current_confidence:.2f}")
                print(f"FPS: {current_fps:.1f}")
                print(f"Буфер кадров: {len(self.keypoints_buffer)}/{self.sequence_length}")
                print(f"Стабильность трекинга: {self.stable_hands_count}/{self.min_stable_frames}")
                print(f"Стабильность жеста: {self.gesture_stability_counter}/{self.min_gesture_stability}")
                print(f"Последние предсказания: {list(self.prediction_buffer)[-5:] if len(self.prediction_buffer) >= 5 else list(self.prediction_buffer)}")
                print(f"Всего классов: {len(self.class_to_idx)}")
                print(f"==================\n")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

def main():
    """Основная функция"""
    model_path = 'D:/gesture/13class_model_output/13class_gesture_model.pth'
    config_path = 'D:/gesture/13class_model_output/config.json'
    
    # Проверяем существование файлов
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл модели не найден: {model_path}")
        print("Сначала запустите train_13class_model.py для обучения модели")
        return
    
    if not os.path.exists(config_path):
        print(f"Ошибка: Файл конфигурации не найден: {config_path}")
        return
    
    try:
        recognizer = ThirteenClassGestureRecognizer(model_path, config_path)
        recognizer.run()
    except Exception as e:
        print(f"Ошибка при запуске распознавания: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()