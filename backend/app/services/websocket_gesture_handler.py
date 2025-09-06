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
import time
import logging
import pickle

# Добавляем пути к модулям
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from realtime_13class_enhanced_recognition import EnhancedGestureRecognizer

class WebSocketGestureHandler:
    """Обработчик жестов для WebSocket с использованием EnhancedGestureRecognizer"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Пути к модели и конфигурации
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'clean_model'))
        model_path = os.path.join(base_path, '13class_model_output', '13class_gesture_model.pth')
        config_path = os.path.join(base_path, '13class_model_output', 'config.json')
        
        self.logger.info(f"Пути к файлам: model={model_path}, config={config_path}")
        self.logger.info(f"Существование файлов: model={os.path.exists(model_path)}, config={os.path.exists(config_path)}")
        
        try:
            # Инициализация Enhanced Gesture Recognizer
            self.recognizer = EnhancedGestureRecognizer(model_path, config_path)
            self.logger.info("Enhanced Gesture Recognizer инициализирован успешно")
            self.is_initialized = True
        except Exception as e:
            self.logger.error(f"Ошибка инициализации Enhanced Gesture Recognizer: {e}")
            self.is_initialized = False
            self.recognizer = None
        
        # Параметры для WebSocket обработки
        self.last_prediction_time = 0
        self.prediction_cooldown = 0.1  # 100мс между предсказаниями
        
    def process_frame_data(self, frame_data: str, model_type: str = '13class') -> Dict[str, Any]:
        """Обработка данных кадра от WebSocket клиента"""
        if not self.is_initialized:
            return {
                'success': False,
                'error': 'Gesture recognizer not initialized',
                'gesture': 'no_event',
                'confidence': 0.0
            }
        
        try:
            # Декодирование base64 изображения
            if frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]
            
            image_bytes = base64.b64decode(frame_data)
            image = Image.open(BytesIO(image_bytes))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Проверка cooldown
            current_time = time.time()
            if current_time - self.last_prediction_time < self.prediction_cooldown:
                return {
                'success': True,
                'prediction': getattr(self.recognizer, 'current_gesture', 'Ожидание...'),
                'confidence': getattr(self.recognizer, 'current_confidence', 0.0),
                'cooldown': True
            }
            
            # Извлечение ключевых точек
            keypoints, hands_results, pose_results, hands_detected, pose_detected = self.recognizer.extract_enhanced_keypoints(frame)
            
            # Добавление в буфер
            self.recognizer.keypoints_buffer.append(keypoints)
            
            # Предсказание жеста если буфер заполнен
            gesture = 'no_event'
            confidence = 0.0
            
            if len(self.recognizer.keypoints_buffer) >= self.recognizer.sequence_length:
                sequence = list(self.recognizer.keypoints_buffer)
                predicted_class, pred_confidence = self.recognizer.predict_gesture(sequence)
                
                if predicted_class is not None:
                    # Фильтрация предсказаний
                    filtered_gesture, filtered_confidence = self.recognizer.filter_predictions(predicted_class, pred_confidence)
                    gesture = filtered_gesture
                    confidence = filtered_confidence
                    
                    # Обновляем состояние recognizer
                    self.recognizer.current_gesture = gesture
                    self.recognizer.current_confidence = confidence
            
            self.last_prediction_time = current_time
            
            return {
                'success': True,
                'prediction': gesture,
                'confidence': float(confidence),
                'hands_detected': hands_detected,
                'pose_detected': pose_detected,
                'buffer_length': len(self.recognizer.keypoints_buffer)
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки кадра: {e}")
            return {
                'success': False,
                'error': str(e),
                'prediction': 'Ошибка',
                'confidence': 0.0
            }
    
    def reset_sequence(self, model_type: str = '13class'):
        """Сброс последовательности жестов"""
        if self.is_initialized and hasattr(self.recognizer, 'keypoints_buffer'):
            self.recognizer.keypoints_buffer.clear()
            self.recognizer.prediction_buffer.clear()
            self.recognizer.current_gesture = "Ожидание..."
            self.recognizer.current_confidence = 0.0
            self.logger.info("Последовательность жестов сброшена")
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус обработчика жестов"""
        return {
            'is_initialized': self.is_initialized,
            'model_loaded': self.recognizer is not None,
            'sequence_length': len(self.recognizer.keypoints_buffer) if self.is_initialized and hasattr(self.recognizer, 'keypoints_buffer') else 0,
            'current_gesture': getattr(self.recognizer, 'current_gesture', 'N/A') if self.is_initialized else 'N/A',
            'current_confidence': getattr(self.recognizer, 'current_confidence', 0.0) if self.is_initialized else 0.0,
            'device': str(self.recognizer.device) if self.is_initialized else 'N/A',
            'timestamp': time.time()
        }

# Глобальный экземпляр обработчика
websocket_gesture_handler = WebSocketGestureHandler()