import React, { useState, useRef, useEffect } from 'react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { ScrollArea } from '../components/ui/scroll-area';
import { Badge } from '../components/ui/badge';
import { Separator } from '../components/ui/separator';
import { Camera, Send, Square, RotateCcw, Mic, MicOff } from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  gesture?: {
    prediction: string;
    confidence: number;
  };
}

interface GestureResult {
  prediction: string;
  confidence: number;
  class_id?: number;
  timestamp?: string;
  error?: string;
}

const GestureChat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'system',
      content: 'Добро пожаловать в чат с распознаванием жестов! Включите камеру для начала работы.',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [currentGesture, setCurrentGesture] = useState<GestureResult | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Инициализация WebSocket соединения
  const initWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8000/api/v1/gesture/ws/recognize');
      
      wsRef.current.onopen = () => {
        console.log('WebSocket соединение установлено');
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log('Получено WebSocket сообщение:', message);
          
          if (message.type === 'gesture_result') {
            const result: GestureResult = message.data;
            setCurrentGesture(result);
            
            // Если жест распознан с высокой уверенностью, добавляем сообщение
            if (result.confidence > 0.7 && 
                result.prediction !== 'Накопление данных...' && 
                result.prediction !== 'Ожидание...' && 
                result.prediction !== 'Руки не обнаружены' &&
                result.prediction !== 'no_event') {
              addMessage({
                type: 'system',
                content: `Распознан жест: ${result.prediction} (уверенность: ${(result.confidence * 100).toFixed(1)}%)`,
                gesture: {
                  prediction: result.prediction,
                  confidence: result.confidence
                }
              });
            }
          } else if (message.type === 'connection_status') {
            console.log('Статус подключения:', message.data);
          } else if (message.type === 'error') {
            console.error('Ошибка от сервера:', message.data.message);
          } else if (message.type === 'pong') {
            console.log('Получен pong от сервера');
          }
        } catch (error) {
          console.error('Ошибка парсинга WebSocket сообщения:', error);
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket ошибка:', error);
      };
      
      wsRef.current.onclose = () => {
        console.log('WebSocket соединение закрыто');
      };
    } catch (error) {
      console.error('Ошибка инициализации WebSocket:', error);
    }
  };

  // Запуск камеры
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          facingMode: 'user'
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setCameraActive(true);
        
        // Инициализируем WebSocket
        initWebSocket();
        
        // Запускаем отправку кадров
        startFrameCapture();
      }
    } catch (error) {
      console.error('Ошибка доступа к камере:', error);
      addMessage({
        type: 'system',
        content: 'Ошибка доступа к камере. Проверьте разрешения.'
      });
    }
  };

  // Остановка камеры
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    setCameraActive(false);
    setCurrentGesture(null);
  };

  // Захват и отправка кадров
  const startFrameCapture = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    intervalRef.current = setInterval(() => {
      if (videoRef.current && canvasRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        const ctx = canvas.getContext('2d');
        
        if (ctx) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0);
          
          // Конвертируем в base64
          const frameData = canvas.toDataURL('image/jpeg', 0.8);
          
          // Отправляем через WebSocket в новом формате
          wsRef.current?.send(JSON.stringify({
            type: 'frame',
            data: frameData
          }));
        }
      }
    }, 200); // Отправляем кадры каждые 200мс
  };

  // Добавление сообщения
  const addMessage = (message: Omit<Message, 'id' | 'timestamp'>) => {
    const newMessage: Message = {
      ...message,
      id: Date.now().toString(),
      timestamp: new Date()
    };
    setMessages(prev => [...prev, newMessage]);
  };

  // Отправка текстового сообщения
  const sendMessage = async () => {
    if (!inputMessage.trim()) return;
    
    const userMessage = inputMessage.trim();
    setInputMessage('');
    
    // Добавляем сообщение пользователя
    addMessage({
      type: 'user',
      content: userMessage
    });
    
    setIsLoading(true);
    
    try {
      // Здесь будет интеграция с AI чатом
      // Пока что добавляем заглушку
      setTimeout(() => {
        addMessage({
          type: 'assistant',
          content: `Вы написали: "${userMessage}". Это демо-ответ от AI ассистента. Интеграция с реальным AI будет добавлена позже.`
        });
        setIsLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Ошибка отправки сообщения:', error);
      addMessage({
        type: 'system',
        content: 'Ошибка отправки сообщения. Попробуйте еще раз.'
      });
      setIsLoading(false);
    }
  };

  // Сброс последовательности жестов
  const resetGestureSequence = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'reset'
      }));
      setCurrentGesture(null);
      addMessage({
        type: 'system',
        content: 'Последовательность жестов сброшена'
      });
    }
  };

  // Ping для поддержания соединения
  const sendPing = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'ping',
        timestamp: Date.now()
      }));
    }
  };

  // Сброс чата
  const resetChat = () => {
    setMessages([
      {
        id: '1',
        type: 'system',
        content: 'Чат сброшен. Добро пожаловать!',
        timestamp: new Date()
      }
    ]);
    setCurrentGesture(null);
  };

  // Периодический ping для поддержания соединения
  useEffect(() => {
    let pingInterval: NodeJS.Timeout;
    
    if (cameraActive && wsRef.current?.readyState === WebSocket.OPEN) {
      pingInterval = setInterval(() => {
        sendPing();
      }, 30000); // Ping каждые 30 секунд
    }
    
    return () => {
      if (pingInterval) {
        clearInterval(pingInterval);
      }
    };
  }, [cameraActive]);

  // Очистка при размонтировании
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Чат с распознаванием жестов
          </h1>
          <p className="text-lg text-gray-600">
            Общайтесь с AI ассистентом используя текст и жесты
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Панель камеры и распознавания */}
          <div className="lg:col-span-1">
            <Card className="h-fit">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Camera className="w-5 h-5" />
                  Распознавание жестов
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Видео поток */}
                <div className="relative">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-48 bg-gray-200 rounded-lg object-cover"
                    style={{ display: cameraActive ? 'block' : 'none' }}
                  />
                  {!cameraActive && (
                    <div className="w-full h-48 bg-gray-200 rounded-lg flex items-center justify-center">
                      <p className="text-gray-500">Камера выключена</p>
                    </div>
                  )}
                  <canvas ref={canvasRef} style={{ display: 'none' }} />
                </div>

                {/* Управление камерой */}
                <div className="flex gap-2">
                  {!cameraActive ? (
                    <Button onClick={startCamera} className="flex-1">
                      <Camera className="w-4 h-4 mr-2" />
                      Включить камеру
                    </Button>
                  ) : (
                    <Button onClick={stopCamera} variant="destructive" className="flex-1">
                      <Square className="w-4 h-4 mr-2" />
                      Остановить
                    </Button>
                  )}
                </div>
                
                {/* Дополнительные кнопки управления */}
                {cameraActive && (
                  <div className="flex gap-2">
                    <Button onClick={resetGestureSequence} variant="outline" className="flex-1">
                      <RotateCcw className="w-4 h-4 mr-2" />
                      Сбросить
                    </Button>
                  </div>
                )}

                {/* Текущий жест */}
                {currentGesture && (
                  <div className="p-3 bg-blue-50 rounded-lg">
                    <h4 className="font-semibold text-sm mb-2">Текущий жест:</h4>
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{currentGesture.prediction}</span>
                      <Badge variant="secondary">
                        {currentGesture.confidence && !isNaN(currentGesture.confidence) 
                          ? (currentGesture.confidence * 100).toFixed(1) + '%'
                          : '0.0%'
                        }
                      </Badge>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Чат */}
          <div className="lg:col-span-2">
            <Card className="h-[600px] flex flex-col">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Чат с AI ассистентом</CardTitle>
                  <Button onClick={resetChat} variant="outline" size="sm">
                    <RotateCcw className="w-4 h-4 mr-2" />
                    Сбросить
                  </Button>
                </div>
              </CardHeader>
              
              <CardContent className="flex-1 flex flex-col">
                {/* Сообщения */}
                <ScrollArea className="flex-1 pr-4">
                  <div className="space-y-4">
                    {messages.map((message) => (
                      <div key={message.id} className="flex flex-col">
                        <div className={`flex ${
                          message.type === 'user' ? 'justify-end' : 'justify-start'
                        }`}>
                          <div className={`max-w-[80%] p-3 rounded-lg ${
                            message.type === 'user'
                              ? 'bg-blue-500 text-white'
                              : message.type === 'assistant'
                              ? 'bg-gray-100 text-gray-900'
                              : 'bg-yellow-50 text-yellow-800 border border-yellow-200'
                          }`}>
                            <p className="text-sm">{message.content}</p>
                            {message.gesture && (
                              <div className="mt-2 pt-2 border-t border-opacity-20">
                                <Badge variant="secondary" className="text-xs">
                                  Жест: {message.gesture.prediction}
                                </Badge>
                              </div>
                            )}
                          </div>
                        </div>
                        <div className={`text-xs text-gray-500 mt-1 ${
                          message.type === 'user' ? 'text-right' : 'text-left'
                        }`}>
                          {message.timestamp.toLocaleTimeString()}
                        </div>
                      </div>
                    ))}
                    {isLoading && (
                      <div className="flex justify-start">
                        <div className="bg-gray-100 p-3 rounded-lg">
                          <div className="flex items-center space-x-2">
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                            <span className="text-sm text-gray-600">AI думает...</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </ScrollArea>

                <Separator className="my-4" />

                {/* Ввод сообщения */}
                <div className="flex gap-2">
                  <Input
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    placeholder="Введите сообщение..."
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    disabled={isLoading}
                  />
                  <Button onClick={sendMessage} disabled={isLoading || !inputMessage.trim()}>
                    <Send className="w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GestureChat;