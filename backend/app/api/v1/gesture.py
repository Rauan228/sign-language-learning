from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import asyncio
import logging
from app.services.gesture_recognition import gesture_service
from app.services.websocket_gesture_handler import websocket_gesture_handler

router = APIRouter()

class GestureRequest(BaseModel):
    frame_data: str
    model_type: Optional[str] = '13class'

class GestureResponse(BaseModel):
    prediction: str
    confidence: float
    class_id: Optional[int] = None
    all_probabilities: Optional[list] = None
    error: Optional[str] = None

@router.post("/recognize", response_model=GestureResponse)
async def recognize_gesture(request: GestureRequest):
    """
    Распознавание жеста по изображению
    
    - **frame_data**: Base64 строка изображения
    - **model_type**: Тип модели ('13class' или другие доступные)
    """
    try:
        result = gesture_service.process_frame(request.frame_data, request.model_type)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return GestureResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")

@router.post("/reset/{model_type}")
async def reset_gesture_sequence(model_type: str = '13class'):
    """
    Сброс последовательности для указанной модели
    
    - **model_type**: Тип модели для сброса
    """
    try:
        gesture_service.reset_sequence(model_type)
        return {"message": f"Последовательность для модели {model_type} сброшена"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")

@router.get("/models")
async def get_available_models():
    """
    Получение списка доступных моделей
    """
    try:
        models = list(gesture_service.models.keys())
        configs = {}
        
        for model_type in models:
            config = gesture_service.configs.get(model_type, {})
            configs[model_type] = {
                'num_classes': config.get('num_classes', 0),
                'class_names': config.get('class_names', {}),
                'sequence_length': config.get('sequence_length', 0)
            }
        
        return {
            'available_models': models,
            'model_configs': configs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")

# WebSocket для real-time распознавания
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/recognize")
async def websocket_gesture_recognition(websocket: WebSocket):
    """
    WebSocket endpoint для real-time распознавания жестов с улучшенной обработкой
    
    Ожидает JSON сообщения в формате:
    {
        "type": "frame",
        "data": "base64_image_string"
    }
    
    Отправляет JSON ответы в формате:
    {
        "type": "gesture_result",
        "data": {
            "prediction": "gesture_name",
            "confidence": 0.95,
            "class_id": 1,
            "stable_tracking": true
        }
    }
    """
    await manager.connect(websocket)
    logger = logging.getLogger(__name__)
    logger.info("WebSocket connection established")
    
    try:
        # Отправляем статус подключения
        status_response = {
            'type': 'connection_status',
            'data': {
                'status': 'connected',
                'handler_status': websocket_gesture_handler.get_status()
            }
        }
        await manager.send_personal_message(
            json.dumps(status_response, ensure_ascii=False),
            websocket
        )
        
        while True:
            # Получение данных от клиента
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                logger.debug(f"Received message type: {message.get('type', 'unknown')}")
                
                if message.get('type') == 'frame':
                    # Обрабатываем кадр с помощью нового обработчика
                    frame_data = message.get('data')
                    if frame_data:
                        # Используем новый WebSocket обработчик
                        result = websocket_gesture_handler.process_frame_data(frame_data)
                        
                        # Отправляем результат обратно клиенту
                        response = {
                            'type': 'gesture_result',
                            'data': result
                        }
                        await manager.send_personal_message(
                            json.dumps(response, ensure_ascii=False),
                            websocket
                        )
                        logger.debug(f"Sent result: {result.get('prediction', 'unknown')}")
                    else:
                        logger.warning("Received empty frame data")
                        await manager.send_personal_message(
                            json.dumps({
                                'type': 'error',
                                'data': {'message': 'Отсутствуют данные кадра'}
                            }),
                            websocket
                        )
                
                elif message.get('type') == 'reset':
                    # Сбрасываем последовательность
                    websocket_gesture_handler.reset_sequence()
                    
                    response = {
                        'type': 'reset_complete',
                        'data': {'status': 'success'}
                    }
                    await manager.send_personal_message(
                        json.dumps(response, ensure_ascii=False),
                        websocket
                    )
                    logger.info("Sequence reset completed")
                
                elif message.get('type') == 'ping':
                    # Ответ на ping для поддержания соединения
                    response = {
                        'type': 'pong',
                        'data': {'timestamp': message.get('timestamp', 0)}
                    }
                    await manager.send_personal_message(
                        json.dumps(response, ensure_ascii=False),
                        websocket
                    )
                
                else:
                    logger.warning(f"Unknown message type: {message.get('type')}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                await manager.send_personal_message(
                    json.dumps({
                        'type': 'error',
                        'data': {'message': 'Неверный формат JSON'}
                    }),
                    websocket
                )
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                import traceback
                traceback.print_exc()
                await manager.send_personal_message(
                    json.dumps({
                        'type': 'error',
                        'data': {'message': f'Ошибка обработки: {str(e)}'}
                    }),
                    websocket
                )
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed by client")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        manager.disconnect(websocket)