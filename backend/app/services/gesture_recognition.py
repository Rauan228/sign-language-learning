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
import base64
from io import BytesIO
from PIL import Image
import asyncio
from typing import Optional, Dict, Any
import joblib

# Импортируем модель из clean_model
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'clean_model'))
from gesture_lstm_model import GestureLSTM

class GestureRecognitionService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Загружаем модели
        self.models = {}
        self.configs = {}
        self.scalers = {}
        self.sequences = {}
        
        # Параметры для стабилизации трекинга
        self.stable_hands_count = 0
        self.min_stable_frames = 3
        self.last_keypoints = None
        
        # Параметры для фильтрации предсказаний
        self.confidence_threshold = {
            'no_event': 0.6,
            'gesture': 0.7
        }
        self.gesture_stability_counter = 0
        self.last_stable_gesture = None
        self.current_gesture = 'no_event'
        self.current_confidence = 0.0
        self.prediction_buffer = deque(maxlen=5)
        
        self._load_models()
    
    def _load_models(self):
        """Загрузка моделей распознавания жестов"""
        base_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
        
        # Загрузка 13-классовой модели
        model_13_path = os.path.join(base_path, '13class_model_output', '13class_gesture_model.pth')
        config_13_path = os.path.join(base_path, '13class_model_output', 'config.json')
        
        if os.path.exists(model_13_path) and os.path.exists(config_13_path):
            with open(config_13_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            model = GestureLSTM(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                num_classes=config['num_classes'],
                dropout=config.get('dropout', 0.2)
            )
            
            checkpoint = torch.load(model_13_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.models['13class'] = model
            self.configs['13class'] = config
            # Используем дефолтное значение для sequence_length если его нет в конфиге
            sequence_length = config.get('sequence_length', 30)
            self.sequences['13class'] = deque(maxlen=sequence_length)
            
            # Загрузка скейлера
            scaler_path = os.path.join(base_path, '13class_model_output', 'scaler.pkl')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                self.scalers['13class'] = scaler
            else:
                # Fallback: создаем новый скейлер
                scaler = StandardScaler()
                dummy_data = np.random.randn(100, config['input_size'])
                scaler.fit(dummy_data)
                self.scalers['13class'] = scaler
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Извлечение ключевых точек из кадра с оптимизацией и стабилизацией"""
        try:
            # Уменьшаем размер кадра для ускорения обработки
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_small = cv2.resize(frame, (new_width, new_height))
            else:
                frame_small = frame
                
            rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            
            # Обработка рук
            hands_results = self.hands.process(rgb_frame)
            
            # Обработка позы
            pose_results = self.pose.process(rgb_frame)
            
            keypoints = []
            hands_detected = 0
            
            # Извлечение ключевых точек рук с улучшенной стабилизацией
            if hands_results.multi_hand_landmarks:
                hands_detected = len(hands_results.multi_hand_landmarks)
                
                # Сортируем руки по x-координате для консистентности
                hand_data = []
                for i, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    hand_keypoints = []
                    avg_x = sum(landmark.x for landmark in hand_landmarks.landmark) / len(hand_landmarks.landmark)
                    for landmark in hand_landmarks.landmark:
                        hand_keypoints.extend([landmark.x, landmark.y, landmark.z])
                    hand_data.append((avg_x, hand_keypoints))
                
                # Сортируем по x-координате (левая рука первая)
                hand_data.sort(key=lambda x: x[0])
                
                for _, hand_keypoints in hand_data:
                    keypoints.extend(hand_keypoints)
            
            # Стабилизация трекинга рук
            if hands_detected > 0:
                self.stable_hands_count = min(self.stable_hands_count + 1, self.min_stable_frames)
            else:
                self.stable_hands_count = max(self.stable_hands_count - 1, 0)
                
            # Используем последние стабильные keypoints если трекинг нестабилен
            if self.stable_hands_count < self.min_stable_frames and self.last_keypoints is not None:
                # Используем только часть рук из последних стабильных keypoints
                hands_keypoints = self.last_keypoints[:126] if len(self.last_keypoints) >= 126 else [0.0] * 126
                keypoints = hands_keypoints
            elif len(keypoints) > 0:
                # Сохраняем текущие keypoints как стабильные
                current_hands_keypoints = keypoints[:126] if len(keypoints) >= 126 else keypoints + [0.0] * (126 - len(keypoints))
                if self.last_keypoints is None:
                    self.last_keypoints = current_hands_keypoints + [0.0] * 132  # добавляем место для позы
                else:
                    self.last_keypoints[:126] = current_hands_keypoints
            
            # Нормализуем руки до фиксированного размера (126 элементов для 2 рук)
            if len(keypoints) == 0:
                keypoints = [0.0] * 63  # Одна рука
            elif len(keypoints) == 63:  # Одна рука - дублируем или добавляем нули
                keypoints.extend([0.0] * 63)
            elif len(keypoints) > 126:  # Больше 2 рук (обрезаем)
                keypoints = keypoints[:126]
            elif len(keypoints) < 126 and len(keypoints) > 63:
                keypoints.extend([0.0] * (126 - len(keypoints)))
            
            # Убеждаемся, что у нас ровно 126 элементов для рук
            hands_keypoints = keypoints[:126] + [0.0] * max(0, 126 - len(keypoints))
            hands_keypoints = hands_keypoints[:63]  # Берем только одну руку для совместимости
            
            # Извлечение ключевых точек позы
            pose_keypoints = []
            if pose_results.pose_landmarks:
                for landmark in pose_results.pose_landmarks.landmark:
                    pose_keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            else:
                # Если поза не обнаружена, добавляем нули
                pose_keypoints.extend([0.0] * 132)  # 33 точки * 4 координаты
            
            # Объединяем keypoints рук и позы
            final_keypoints = hands_keypoints + pose_keypoints
            
            return np.array(final_keypoints, dtype=np.float32)
            
        except Exception as e:
            print(f"Ошибка при извлечении ключевых точек: {e}")
            return None
    
    def predict_gesture(self, keypoints: np.ndarray, model_type: str = '13class') -> Dict[str, Any]:
        """Предсказание жеста по ключевым точкам с улучшенной фильтрацией"""
        try:
            if model_type not in self.models:
                return {'error': f'Модель {model_type} не найдена'}
            
            model = self.models[model_type]
            config = self.configs[model_type]
            scaler = self.scalers[model_type]
            sequence = self.sequences[model_type]
            
            # Нормализация ключевых точек
            keypoints_scaled = scaler.transform(keypoints.reshape(1, -1))[0]
            
            # Добавление в последовательность
            sequence.append(keypoints_scaled)
            
            # Проверяем, достаточно ли данных для предсказания
            sequence_length = len(sequence)
            min_sequence_length = config.get('sequence_length', 30)
            
            if sequence_length < min_sequence_length:
                return {
                    'prediction': 'Накопление данных...',
                    'confidence': 0.0,
                    'frames_collected': sequence_length,
                    'frames_needed': min_sequence_length
                }
            
            # Подготовка данных для модели с оптимизацией памяти
            sequence_array = np.array(list(sequence), dtype=np.float32)
            sequence_tensor = torch.FloatTensor(sequence_array).unsqueeze(0)
            
            if self.device.type == 'cuda':
                sequence_tensor = sequence_tensor.to(self.device, non_blocking=True)
            else:
                sequence_tensor = sequence_tensor.to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = model(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                predicted_class = predicted_class.item()
                confidence_value = confidence.item()
                
                # Очистка GPU памяти
                if self.device.type == 'cuda':
                    del sequence_tensor, outputs
                    torch.cuda.empty_cache()
            
            # Применяем фильтрацию предсказаний
            filtered_gesture, filtered_confidence = self._filter_predictions(
                predicted_class, confidence_value, model_type
            )
            
            # Получение названия жеста
            raw_gesture_name = config['class_names'].get(str(predicted_class), f'Класс {predicted_class}')
            
            return {
                'prediction': filtered_gesture,
                'confidence': float(filtered_confidence),
                'raw_prediction': raw_gesture_name,
                'raw_confidence': float(confidence_value),
                'class_id': predicted_class,
                'all_probabilities': probabilities[0].cpu().numpy().tolist()
            }
            
        except Exception as e:
            return {
                'prediction': 'Ошибка распознавания',
                'confidence': 0.0,
                'error': f'Ошибка при предсказании: {str(e)}'
            }
    
    def _filter_predictions(self, predicted_class: int, confidence: float, model_type: str) -> tuple:
        """Улучшенная фильтрация предсказаний для стабильности"""
        if predicted_class is None or confidence is None or np.isnan(confidence):
            return self.current_gesture, max(0.0, self.current_confidence)
        
        config = self.configs[model_type]
        gesture_name = config['class_names'].get(str(predicted_class), f'Класс {predicted_class}')
        
        # Применяем разные пороги уверенности
        if gesture_name == 'no_event' or 'no_event' in gesture_name.lower():
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
        
        # Требуем минимум 3 стабильных предсказания для смены жеста
        min_stability = 3
        if self.gesture_stability_counter >= min_stability:
            # Вычисляем среднюю уверенность из буфера для текущего жеста
            current_gesture_predictions = [
                conf for gest, conf in self.prediction_buffer 
                if gest == gesture_name
            ]
            
            if current_gesture_predictions:
                avg_confidence = sum(current_gesture_predictions) / len(current_gesture_predictions)
                self.current_gesture = gesture_name
                self.current_confidence = avg_confidence
                return gesture_name, avg_confidence
        
        return self.current_gesture, self.current_confidence
    
    def get_gesture_statistics(self) -> Dict[str, Any]:
        """Получение статистики распознавания жестов"""
        return {
            'current_gesture': self.current_gesture,
            'current_confidence': self.current_confidence,
            'gesture_stability_counter': self.gesture_stability_counter,
            'stable_hands_count': self.stable_hands_count,
            'prediction_buffer_size': len(self.prediction_buffer),
            'last_stable_gesture': self.last_stable_gesture,
            'confidence_thresholds': self.confidence_threshold
        }
    
    def update_confidence_thresholds(self, gesture_threshold: float = None, no_event_threshold: float = None):
        """Обновление порогов уверенности для фильтрации"""
        if gesture_threshold is not None:
            self.confidence_threshold['gesture'] = gesture_threshold
        if no_event_threshold is not None:
            self.confidence_threshold['no_event'] = no_event_threshold
    
    def process_frame(self, frame_data: str, model_type: str = '13class') -> Dict[str, Any]:
        """Обработка кадра из base64 строки"""
        try:
            # Декодирование base64 изображения
            image_data = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
            image = Image.open(BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Извлечение ключевых точек
            keypoints = self.extract_keypoints(frame)
            
            if keypoints is None:
                return {
                    'prediction': 'Руки не обнаружены',
                    'confidence': 0.0,
                    'error': 'Не удалось извлечь ключевые точки'
                }
            
            # Предсказание жеста
            result = self.predict_gesture(keypoints, model_type)
            
            return result
            
        except Exception as e:
            return {
                'prediction': 'Ошибка обработки',
                'confidence': 0.0,
                'error': f'Ошибка при обработке кадра: {str(e)}'
            }
    
    def reset_sequence(self, model_type: str = '13class'):
        """Сброс последовательности для модели"""
        if model_type in self.sequences:
            self.sequences[model_type].clear()
        
        # Сбрасываем параметры фильтрации
        self.gesture_stability_counter = 0
        self.last_stable_gesture = None
        self.current_gesture = 'no_event'
        self.current_confidence = 0.0
        self.prediction_buffer.clear()
        self.stable_hands_count = 0
        self.last_keypoints = None

# Глобальный экземпляр сервиса
gesture_service = GestureRecognitionService()