# -*- coding: utf-8 -*-
"""
Приложение реального времени для распознавания 13 классов жестов
Основано на clean_model/realtime_all_classes_gesture_recognition.py
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import json
import pickle
from collections import deque
import time
from gesture_lstm_model_13class import GestureLSTM
import sys
from PIL import Image, ImageDraw, ImageFont

# Установка кодировки для корректного отображения кириллицы
sys.stdout.reconfigure(encoding='utf-8')

class GestureRecognizer:
    def __init__(self, model_path, config_path, scaler_path):
        """
        Инициализация распознавателя жестов
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        
        # Загрузка конфигурации
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Загрузка scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
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
        
        print(f"Модель загружена. Точность: {checkpoint['accuracy']:.2f}%")
        print(f"Классы: {list(self.config['idx_to_class'].values())}")
        
        # Инициализация MediaPipe - улучшенные настройки для точного распознавания
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,                # Поддержка двух рук
            min_detection_confidence=0.7,   # Оптимизировано для лучшего детектирования
            min_tracking_confidence=0.5,    # Снижено для более чувствительного трекинга
            model_complexity=1              # Максимальная точность модели
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Параметры обработки - оптимизированы для стабильности
        self.sequence_length = 30
        self.keypoints_buffer = deque(maxlen=self.sequence_length)
        self.prediction_buffer = deque(maxlen=20)  # Увеличен буфер для большей стабильности
        
        # Пороги уверенности - оптимизированы для стабильной работы
        self.no_event_threshold = 0.4   # Снижен для лучшего распознавания
        self.gesture_threshold = 0.75   # Оптимизирован для баланса точности и отзывчивости
        
        # Состояние
        self.current_prediction = "no_event"
        self.prediction_confidence = 0.0
        self.last_prediction_time = 0
        self.prediction_cooldown = 0.5  # секунды
        
        # Счетчики для статистики
        self.frame_count = 0
        self.prediction_count = 0
        self.stable_predictions = 0
        
        # Параметры отображения - замедлено для стабильности
        self.frame_skip = 3  # Обрабатывать каждый N-й кадр
        self.text_update_interval = 0.1  # Обновлять текст каждые 0.1 сек
        self.last_text_update = 0
        self.min_confidence_frames = 5  # Минимум кадров для уверенного предсказания
        
        # Кэш для отображения
        self.text_cache = {
            'prediction': 'no_event',
            'confidence': 0.0,
            'stability': 0
        }
    
    def extract_keypoints(self, frame):
        """
        Извлечение ключевых точек рук из кадра с улучшенной обработкой
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Инициализация массива для ключевых точек (126 признаков)
        keypoints = np.zeros(126)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            # Сортировка рук по позиции (левая/правая)
            hands_data = []
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Определение типа руки
                hand_label = handedness.classification[0].label  # 'Left' или 'Right'
                hand_score = handedness.classification[0].score
                
                # Извлечение координат
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    hand_points.extend([landmark.x, landmark.y, landmark.z])
                
                hands_data.append({
                    'label': hand_label,
                    'score': hand_score,
                    'points': hand_points,
                    'landmarks': hand_landmarks
                })
            
            # Сортировка по уверенности детектирования
            hands_data.sort(key=lambda x: x['score'], reverse=True)
            
            # Размещение данных рук в массиве (приоритет более уверенной руке)
            for hand_idx, hand_data in enumerate(hands_data[:2]):
                start_idx = hand_idx * 63
                keypoints[start_idx:start_idx + 63] = hand_data['points']
        
        return keypoints, results
    
    def predict_gesture(self):
        """
        Предсказание жеста на основе накопленных ключевых точек
        """
        if len(self.keypoints_buffer) < self.sequence_length:
            return "no_event", 0.0
        
        # Подготовка данных
        sequence = np.array(list(self.keypoints_buffer))
        sequence = sequence.reshape(1, self.sequence_length, -1)
        
        # Нормализация
        sequence_reshaped = sequence.reshape(-1, sequence.shape[-1])
        sequence_normalized = self.scaler.transform(sequence_reshaped)
        sequence_normalized = sequence_normalized.reshape(sequence.shape)
        
        # Предсказание
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence_normalized).to(self.device)
            output = self.model(sequence_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = self.config['idx_to_class'][str(predicted_idx.item())]
            confidence_value = confidence.item()
        
        return predicted_class, confidence_value
    
    def filter_predictions(self, prediction, confidence):
        """
        Фильтрация предсказаний для стабильности с улучшенной стабильностью
        """
        current_time = time.time()
        
        # Добавление в буфер предсказаний
        self.prediction_buffer.append((prediction, confidence))
        
        # Проверка кулдауна
        if current_time - self.last_prediction_time < self.prediction_cooldown:
            return self.current_prediction, self.prediction_confidence
        
        # Анализ последних предсказаний с улучшенной фильтрацией
        if len(self.prediction_buffer) >= 6:
            recent_predictions = list(self.prediction_buffer)[-12:]  # Анализ последних 12 кадров
            
            # Подсчет вхождений каждого класса с весами по уверенности
            prediction_scores = {}
            for pred, conf in recent_predictions:
                if pred not in prediction_scores:
                    prediction_scores[pred] = []
                prediction_scores[pred].append(conf)
            
            # Поиск наиболее стабильного предсказания
            best_prediction = None
            best_score = 0.0
            best_confidence = 0.0
            
            for pred, confs in prediction_scores.items():
                # Вычисляем взвешенный счет: количество * средняя уверенность
                count = len(confs)
                avg_conf = np.mean(confs)
                stability = np.std(confs) if len(confs) > 1 else 0  # Стабильность (меньше = лучше)
                
                # Бонус за стабильность (меньший разброс)
                stability_bonus = max(0, 1.0 - stability)
                weighted_score = count * avg_conf * stability_bonus
                
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_prediction = pred
                    best_confidence = avg_conf
            
            # Требуем минимальное количество голосов для стабильности
            min_votes_required = max(3, len(recent_predictions) // 4)
            pred_count = len(prediction_scores.get(best_prediction, []))
            
            if pred_count >= min_votes_required:
                # Применение адаптивных порогов уверенности
                if best_prediction == 'no_event':
                    threshold = self.no_event_threshold
                else:
                    # Динамический порог в зависимости от стабильности
                    base_threshold = self.gesture_threshold
                    stability_factor = min(pred_count / len(recent_predictions), 1.0)
                    threshold = base_threshold * (1.0 - 0.2 * stability_factor)  # Снижаем порог для стабильных предсказаний
                
                if best_confidence >= threshold:
                    self.current_prediction = best_prediction
                    self.prediction_confidence = best_confidence
                    self.last_prediction_time = current_time
                    self.stable_predictions += 1
        
        return self.current_prediction, self.prediction_confidence
    
    def draw_landmarks(self, frame, results):
        """
        Отрисовка ключевых точек рук с улучшенной визуализацией
        """
        if results.multi_hand_landmarks and results.multi_handedness:
            # Цвета для разных рук
            colors = {
                'Left': (255, 0, 0),   # Синий для левой руки
                'Right': (0, 0, 255)   # Красный для правой руки
            }
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Определение типа руки
                hand_label = handedness.classification[0].label
                hand_score = handedness.classification[0].score
                
                # Выбор цвета в зависимости от руки
                landmark_color = colors.get(hand_label, (0, 255, 0))
                connection_color = tuple(int(c * 0.7) for c in landmark_color)  # Более темный для соединений
                
                # Отрисовка ключевых точек
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=landmark_color, thickness=3, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=connection_color, thickness=2)
                )
                
                # Добавление подписи руки
                if hand_landmarks.landmark:
                    # Получение координат запястья (точка 0)
                    wrist = hand_landmarks.landmark[0]
                    x = int(wrist.x * frame.shape[1])
                    y = int(wrist.y * frame.shape[0]) - 20
                    
                    # Отображение типа руки и уверенности
                    label_text = f"{hand_label} ({hand_score:.2f})"
                    cv2.putText(frame, label_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, landmark_color, 2)
    
    def draw_text_with_pillow(self, frame, text, position, font_size=24, color=(255, 255, 255)):
        """Отрисовка текста с поддержкой кириллицы через Pillow"""
        # Конвертация BGR в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            # Попытка загрузить системный шрифт с поддержкой кириллицы
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                # Альтернативный шрифт
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
            except:
                # Использование стандартного шрифта
                font = ImageFont.load_default()
        
        # Отрисовка текста
        draw.text(position, text, font=font, fill=color)
        
        # Конвертация обратно в BGR
        frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame_bgr
    
    def draw_info(self, frame):
        """
        Отрисовка информации на кадре с поддержкой кириллицы через Pillow
        """
        h, w = frame.shape[:2]
        
        # Фон для текста
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Отрисовка текста с поддержкой кириллицы
        frame = self.draw_text_with_pillow(frame, f"Жест: {self.text_cache['prediction']}", (20, 20), 28, (0, 255, 0))
        frame = self.draw_text_with_pillow(frame, f"Уверенность: {self.text_cache['confidence']:.3f}", (20, 55), 20, (255, 255, 255))
        frame = self.draw_text_with_pillow(frame, f"Стабильность: {self.text_cache['stability']}/20", (20, 85), 20, (255, 255, 255))
        
        # Статус стабильности
        stability_color = (0, 255, 0) if self.text_cache['stability'] >= 8 else (255, 255, 0)
        status_text = "Стабильно" if self.text_cache['stability'] >= 8 else "Стабилизация"
        frame = self.draw_text_with_pillow(frame, f"Статус: {status_text}", (20, 115), 18, stability_color)
        
        # Индикатор состояния
        status_color = (0, 255, 0) if len(self.keypoints_buffer) == self.sequence_length else (0, 0, 255)
        cv2.circle(frame, (w - 30, 30), 10, status_color, -1)
        
        # Статистика буферов (английский текст - можно использовать cv2.putText)
        buffer_info = f"Keypoints: {len(self.keypoints_buffer)}/{self.sequence_length}"
        cv2.putText(frame, buffer_info, (w - 300, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Инструкции с поддержкой кириллицы
        instructions = [
            "q - выход",
            "r - сброс",
            "s - статистика"
        ]
        
        for i, instruction in enumerate(instructions):
            frame = self.draw_text_with_pillow(frame, instruction, (w - 150, h - 60 + i * 25), 16, (200, 200, 200))
        
        return frame
    
    def update_text_cache(self):
        """
        Обновление кэша текста для отображения
        """
        current_time = time.time()
        if current_time - self.last_text_update >= self.text_update_interval:
            self.text_cache['prediction'] = self.current_prediction
            self.text_cache['confidence'] = self.prediction_confidence
            self.text_cache['stability'] = len(self.prediction_buffer)
            self.last_text_update = current_time
    
    def reset_state(self):
        """
        Сброс состояния распознавателя с улучшенной очисткой
        """
        self.keypoints_buffer.clear()
        self.prediction_buffer.clear()
        self.current_prediction = "no_event"
        self.prediction_confidence = 0.0
        self.last_prediction_time = 0
        self.stable_predictions = 0
        
        # Сброс кэша отображения
        self.text_cache = {
            'prediction': 'no_event',
            'confidence': 0.0,
            'stability': 0
        }
        
        print("Состояние полностью сброшено")
    
    def print_statistics(self):
        """
        Вывод статистики
        """
        if self.frame_count > 0:
            stability_rate = (self.stable_predictions / self.prediction_count * 100) if self.prediction_count > 0 else 0
            
            print("\n=== СТАТИСТИКА ===")
            print(f"Обработано кадров: {self.frame_count}")
            print(f"Предсказаний: {self.prediction_count}")
            print(f"Стабильность трекинга: {stability_rate:.1f}%")
            print(f"Текущий жест: {self.current_prediction}")
            print(f"Уверенность: {self.prediction_confidence:.2f}")
            print(f"Размер буфера: {len(self.keypoints_buffer)}/{self.sequence_length}")
            print("==================\n")
    
    def run(self):
        """
        Основной цикл распознавания с улучшенной стабильностью
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Ошибка: Не удалось открыть камеру")
            return
            
        # Настройка камеры для стабильной работы
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Минимальная буферизация
        
        print("Запуск распознавания жестов...")
        print("Управление: 'q' - выход, 'r' - сброс, 's' - статистика")
        
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Ошибка чтения кадра")
                    break
                
                # Отражение кадра для удобства
                frame = cv2.flip(frame, 1)
                
                self.frame_count += 1
                
                # Обработка каждого N-го кадра для стабильности
                if self.frame_count % self.frame_skip == 0:
                    # Извлечение ключевых точек
                    keypoints, results = self.extract_keypoints(frame)
                    self.keypoints_buffer.append(keypoints)
                    
                    # Предсказание жеста
                    if len(self.keypoints_buffer) == self.sequence_length:
                        prediction, confidence = self.predict_gesture()
                        self.prediction_count += 1
                        
                        # Фильтрация предсказаний
                        filtered_prediction, filtered_confidence = self.filter_predictions(prediction, confidence)
                    
                    # Отрисовка ключевых точек
                    self.draw_landmarks(frame, results)
                
                # Обновление текстового кэша
                self.update_text_cache()
                
                # Отрисовка информации
                frame = self.draw_info(frame)
                
                # Подсчет FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    current_fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                
                # Отображение FPS
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Отображение кадра
                cv2.imshow('Распознавание жестов - 13 классов', frame)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Завершение работы...")
                    break
                elif key == ord('r'):
                    self.reset_state()
                    print("Состояние системы сброшено")
                elif key == ord('s'):
                    self.print_statistics()
        
        except KeyboardInterrupt:
            print("\nПрерывание пользователем")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("Распознавание завершено")

def main():
    """Основная функция с улучшенной стабильностью"""
    # Пути к файлам модели
    model_path = "D:/gesture/13class_model_output/13class_gesture_model.pth"
    config_path = "D:/gesture/13class_model_output/config.json"
    scaler_path = "D:/gesture/13class_model_output/scaler.pkl"
    
    print("Инициализация системы распознавания жестов...")
    
    try:
        # Создание и запуск распознавателя
        recognizer = GestureRecognizer(model_path, config_path, scaler_path)
        print("Система готова к работе. Покажите жест руки...")
        recognizer.run()
        
    except FileNotFoundError as e:
        print(f"Ошибка: файл не найден - {e}")
        print("Убедитесь, что модель обучена и файлы находятся в правильной директории")
    except KeyboardInterrupt:
        print("\nПрерывание пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        print("Система распознавания жестов завершена")

if __name__ == '__main__':
    main()