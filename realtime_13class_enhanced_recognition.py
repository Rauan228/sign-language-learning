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

class EnhancedGestureRecognizer:
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
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
            model_complexity=0
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        
        # Параметры оптимизации
        self.frame_skip = 2
        self.frame_counter = 0
        self.processing_frame = None
        self.last_keypoints = None
        self.stable_hands_count = 0
        self.stable_pose_count = 0
        self.min_stable_frames = 3
        
        # Буфер для последовательности кадров
        self.sequence_length = 30
        self.keypoints_buffer = deque(maxlen=self.sequence_length)
        
        # Параметры для фильтрации предсказаний (как в оригинале)
        self.prediction_buffer = deque(maxlen=10)
        self.confidence_threshold = {
            'no_event': 0.6,  # Более низкий порог для no_event
            'gesture': 0.7    # Более высокий порог для жестов
        }
        
        # Состояние распознавания
        self.current_gesture = "Ожидание..."
        self.current_confidence = 0.0
        self.last_prediction_time = time.time()
        self.prediction_cooldown = 0.5  # Как в оригинале
        
        # Кэш для текста (оптимизация отрисовки)
        self.text_cache = {}
        self.last_text_update = 0
        self.text_update_interval = 0.1  # Как в оригинале
        
        print(f"Модель загружена. Классы: {list(self.class_to_idx.keys())}")
        print(f"Лучшая точность модели: {self.config.get('best_accuracy', 'N/A'):.2f}%")
        print(f"Улучшенные признаки: {self.config.get('feature_description', 'Стандартные')}")
    
    def extract_enhanced_keypoints(self, frame):
        """Извлечение ключевых точек рук и позы с оптимизацией"""
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
        
        # Обработка рук
        hands_results = self.hands.process(frame_rgb)
        hand_keypoints = []
        hands_detected = 0
        
        if hands_results.multi_hand_landmarks:
            hands_detected = len(hands_results.multi_hand_landmarks)
            
            # Сортируем руки по x-координате для консистентности
            hand_data = []
            for hand_landmarks in hands_results.multi_hand_landmarks:
                hand_kp = []
                avg_x = sum(landmark.x for landmark in hand_landmarks.landmark) / len(hand_landmarks.landmark)
                for landmark in hand_landmarks.landmark:
                    hand_kp.extend([landmark.x, landmark.y, landmark.z])
                hand_data.append((avg_x, hand_kp))
            
            # Сортируем по x-координате (левая рука первая)
            hand_data.sort(key=lambda x: x[0])
            
            for _, hand_kp in hand_data:
                hand_keypoints.extend(hand_kp)
        
        # Стабилизация трекинга рук
        if hands_detected > 0:
            self.stable_hands_count = min(self.stable_hands_count + 1, self.min_stable_frames)
        else:
            self.stable_hands_count = max(self.stable_hands_count - 1, 0)
        
        # Нормализуем данные рук до 126 элементов
        if len(hand_keypoints) == 0:
            hand_keypoints = [0.0] * 126
        elif len(hand_keypoints) == 63:
            hand_keypoints.extend([0.0] * 63)
        elif len(hand_keypoints) > 126:
            hand_keypoints = hand_keypoints[:126]
        elif len(hand_keypoints) < 126 and len(hand_keypoints) > 63:
            hand_keypoints.extend([0.0] * (126 - len(hand_keypoints)))
        
        hand_keypoints = hand_keypoints[:126] + [0.0] * max(0, 126 - len(hand_keypoints))
        
        # Обработка позы
        pose_results = self.pose.process(frame_rgb)
        pose_keypoints = []
        pose_detected = 0
        
        if pose_results.pose_landmarks:
            pose_detected = 1
            
            # Извлекаем ключевые точки верхней части тела
            important_pose_indices = [
                0,   # нос
                1, 2, 3, 4, 5, 6,  # глаза и уши
                7, 8,  # рот
                9, 10,  # плечи
                11, 12,  # локти
                13, 14,  # запястья
                15, 16,  # бедра
                17, 18,  # колени
                19, 20,  # лодыжки
                21, 22,  # пятки
                23, 24,  # большие пальцы ног
                25, 26,  # мизинцы ног
                27, 28,  # указательные пальцы ног
            ]
            
            for idx in important_pose_indices:
                if idx < len(pose_results.pose_landmarks.landmark):
                    landmark = pose_results.pose_landmarks.landmark[idx]
                    pose_keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                else:
                    pose_keypoints.extend([0.0, 0.0, 0.0, 0.0])
        else:
            pose_keypoints = [0.0] * (29 * 4)
        
        # Стабилизация трекинга позы
        if pose_detected > 0:
            self.stable_pose_count = min(self.stable_pose_count + 1, self.min_stable_frames)
        else:
            self.stable_pose_count = max(self.stable_pose_count - 1, 0)
        
        # Убеждаемся, что у нас ровно 116 элементов для позы
        pose_keypoints = pose_keypoints[:116] + [0.0] * max(0, 116 - len(pose_keypoints))
        
        # Используем последние стабильные keypoints если трекинг нестабилен
        combined_keypoints = hand_keypoints + pose_keypoints
        
        if (self.stable_hands_count < self.min_stable_frames or 
            self.stable_pose_count < self.min_stable_frames) and self.last_keypoints is not None:
            combined_keypoints = self.last_keypoints.copy()
        elif hands_detected > 0 or pose_detected > 0:
            self.last_keypoints = combined_keypoints.copy()
        
        return np.array(combined_keypoints, dtype=np.float32), hands_results, pose_results, hands_detected, pose_detected
    
    def predict_gesture(self, sequence):
        """Предсказание жеста по последовательности с оптимизацией"""
        if len(sequence) < self.sequence_length:
            return None, 0.0
        
        try:
            # Подготовка данных
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
        """Фильтрация предсказаний для стабильности (как в оригинале)"""
        if predicted_class is None:
            return self.current_gesture, self.current_confidence
        
        # Определяем тип предсказания
        gesture_name = self.idx_to_class[predicted_class]
        is_no_event = (gesture_name == 'no_event')
        
        # Применяем соответствующий порог уверенности
        threshold = self.confidence_threshold['no_event'] if is_no_event else self.confidence_threshold['gesture']
        
        if confidence < threshold:
            return self.current_gesture, self.current_confidence
        
        # Добавляем в буфер предсказаний
        self.prediction_buffer.append((predicted_class, confidence))
        
        # Анализируем последние предсказания
        if len(self.prediction_buffer) >= 5:
            recent_predictions = list(self.prediction_buffer)[-5:]
            
            # Подсчитываем частоту каждого класса
            class_counts = {}
            total_confidence = {}
            
            for pred_class, pred_conf in recent_predictions:
                if pred_class not in class_counts:
                    class_counts[pred_class] = 0
                    total_confidence[pred_class] = 0
                class_counts[pred_class] += 1
                total_confidence[pred_class] += pred_conf
            
            # Находим наиболее частый класс
            most_frequent_class = max(class_counts.items(), key=lambda x: x[1])[0]
            
            # Если класс появляется в большинстве предсказаний
            if class_counts[most_frequent_class] >= 3:
                avg_confidence = total_confidence[most_frequent_class] / class_counts[most_frequent_class]
                return self.idx_to_class[most_frequent_class], avg_confidence
        
        return gesture_name, confidence
    
    def run(self):
        """Запуск распознавания в реальном времени с улучшенным трекингом"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Запуск улучшенного распознавания жестов (13 классов + поза)...")
        print("Классы: веселиться, воскресенье, вторник, выходные, любить, осень,")
        print("        понедельник, пятница, редко, среда, суббота, четверг, no_event")
        print("Улучшения: трекинг рук + позы тела для лучшей точности")
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
            
            frame = cv2.flip(frame, 1)
            
            # Пропускаем кадры для оптимизации
            self.frame_counter += 1
            process_frame = (self.frame_counter % self.frame_skip == 0)
            
            if process_frame:
                # Извлекаем улучшенные ключевые точки
                keypoints, hands_results, pose_results, hands_detected, pose_detected = self.extract_enhanced_keypoints(frame)
                
                # Добавляем в буфер только если есть стабильный трекинг
                if (self.stable_hands_count >= self.min_stable_frames or 
                    self.stable_pose_count >= self.min_stable_frames or 
                    hands_detected > 0 or pose_detected > 0):
                    self.keypoints_buffer.append(keypoints)
                
                # Сохраняем результаты для отрисовки
                self.processing_frame = (hands_results, pose_results, hands_detected, pose_detected)
            
            # Рисуем ключевые точки рук
            if self.processing_frame and self.processing_frame[0].multi_hand_landmarks:
                for hand_landmarks in self.processing_frame[0].multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                            color=(255, 255, 255), thickness=2))
            
            # Рисуем ключевые точки позы (только верхняя часть)
            if self.processing_frame and self.processing_frame[1].pose_landmarks:
                # Рисуем только важные соединения верхней части тела
                pose_landmarks = self.processing_frame[1].pose_landmarks
                
                # Плечи и руки
                connections = [
                    (11, 12),  # плечи
                    (11, 13),  # левое плечо-локоть
                    (13, 15),  # левый локоть-запястье
                    (12, 14),  # правое плечо-локоть
                    (14, 16),  # правый локоть-запястье
                    (11, 23),  # левое плечо-бедро
                    (12, 24),  # правое плечо-бедро
                ]
                
                for connection in connections:
                    start_idx, end_idx = connection
                    if (start_idx < len(pose_landmarks.landmark) and 
                        end_idx < len(pose_landmarks.landmark)):
                        start_point = pose_landmarks.landmark[start_idx]
                        end_point = pose_landmarks.landmark[end_idx]
                        
                        if (start_point.visibility > 0.5 and end_point.visibility > 0.5):
                            h, w, _ = frame.shape
                            start_pos = (int(start_point.x * w), int(start_point.y * h))
                            end_pos = (int(end_point.x * w), int(end_point.y * h))
                            cv2.line(frame, start_pos, end_pos, (0, 255, 255), 2)
            
            # Предсказание жеста
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
            
            # Отображение информации
            info_y = 25
            
            # Обновляем текстовый кэш только периодически
            if current_time - self.last_text_update > self.text_update_interval:
                self.text_cache = {
                    'title': "Улучшенное распознавание (13 классов + поза)",
                    'gesture': f"Жест: {self.current_gesture}",
                    'confidence': f"Уверенность: {self.current_confidence:.2f}",
                    'buffer': f"Буфер: {len(self.keypoints_buffer)}/{self.sequence_length}",
                    'hands': f"Руки: {self.processing_frame[2] if self.processing_frame else 0}",
                    'pose': f"Поза: {'Да' if self.processing_frame and self.processing_frame[3] else 'Нет'}",
                    'fps': f"FPS: {current_fps:.1f}",
                    'stability': f"Стабильность: {self.stable_hands_count}/{self.min_stable_frames}"
                }
                self.last_text_update = current_time
            
            # Заголовок
            cv2.putText(frame, self.text_cache['title'], (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            info_y += 25
            
            # Текущий жест
            gesture_color = (0, 255, 0) if self.current_confidence > 0.75 else (0, 165, 255)
            frame = put_russian_text(frame, self.text_cache['gesture'], (10, info_y), 
                                   font_scale=0.6, color=gesture_color, thickness=2)
            info_y += 30
            
            # Уверенность
            cv2.putText(frame, self.text_cache['confidence'], (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, gesture_color, 1)
            info_y += 20
            
            # Статус трекинга
            cv2.putText(frame, self.text_cache['hands'], (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            info_y += 18
            
            cv2.putText(frame, self.text_cache['pose'], (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            info_y += 18
            
            # FPS и стабильность
            cv2.putText(frame, self.text_cache['fps'], (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            info_y += 25
            
            cv2.putText(frame, self.text_cache['stability'], (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            info_y += 30
            
            # Инструкции
            cv2.putText(frame, "'q'-выход 'r'-сброс 's'-статистика", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Индикаторы состояния (компактные)
            # Статус буфера
            buffer_color = (0, 255, 0) if len(self.keypoints_buffer) == self.sequence_length else (0, 0, 255)
            cv2.circle(frame, (frame.shape[1] - 60, 20), 8, buffer_color, -1)
            
            # Статус трекинга
            tracking_color = (0, 255, 0) if self.stable_hands_count >= self.min_stable_frames else (255, 0, 0)
            cv2.circle(frame, (frame.shape[1] - 30, 20), 8, tracking_color, -1)
            
            cv2.imshow('Enhanced Gesture Recognition - 13 Classes + Pose', frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.keypoints_buffer.clear()
                self.prediction_buffer.clear()
                self.current_gesture = "Ожидание..."
                self.current_confidence = 0.0
                self.stable_hands_count = 0
                self.stable_pose_count = 0
                print("Все буферы и счетчики сброшены")
            elif key == ord('s'):
                print(f"\n=== СТАТИСТИКА УЛУЧШЕННОЙ СИСТЕМЫ ===")
                print(f"Текущий жест: {self.current_gesture}")
                print(f"Уверенность: {self.current_confidence:.2f}")
                print(f"FPS: {current_fps:.1f}")
                print(f"Буфер кадров: {len(self.keypoints_buffer)}/{self.sequence_length}")
                print(f"Стабильность трекинга: {self.stable_hands_count}/{self.min_stable_frames}")
                print(f"Размер входных данных: {self.config['input_size']} (242 = 126 рук + 116 поза)")
                print(f"Последние предсказания: {list(self.prediction_buffer)[-5:] if len(self.prediction_buffer) >= 5 else list(self.prediction_buffer)}")
                print(f"Всего классов: {len(self.class_to_idx)}")
                print(f"======================================\n")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.pose.close()

def main():
    """Основная функция"""
    model_path = 'D:/gesture/13class_enhanced_model_output/13class_enhanced_gesture_model.pth'
    config_path = 'D:/gesture/13class_enhanced_model_output/config.json'
    
    # Проверяем существование файлов
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл модели не найден: {model_path}")
        print("Сначала запустите train_13class_enhanced_model.py для обучения модели")
        return
    
    if not os.path.exists(config_path):
        print(f"Ошибка: Файл конфигурации не найден: {config_path}")
        return
    
    try:
        recognizer = EnhancedGestureRecognizer(model_path, config_path)
        recognizer.run()
    except Exception as e:
        print(f"Ошибка при запуске распознавания: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()