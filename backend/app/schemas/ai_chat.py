from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Роли сообщений в чате"""
    user = "user"
    assistant = "assistant"
    system = "system"


class MessageType(str, Enum):
    """Типы сообщений"""
    text = "text"
    image = "image"
    video = "video"
    audio = "audio"
    file = "file"
    gesture = "gesture"


class ChatSessionStatus(str, Enum):
    """Статусы сессий чата"""
    active = "active"
    paused = "paused"
    completed = "completed"
    archived = "archived"


class MessageBase(BaseModel):
    """Базовая схема сообщения"""
    content: str = Field(..., min_length=1, description="Содержимое сообщения")
    role: MessageRole = Field(..., description="Роль отправителя")
    message_type: MessageType = Field(MessageType.text, description="Тип сообщения")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Дополнительные данные")
    attachments: Optional[List[str]] = Field(None, description="Прикрепленные файлы")
    gesture_data: Optional[Dict[str, Any]] = Field(None, description="Данные жестов")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Уровень уверенности AI")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Время обработки в секундах")


class MessageCreate(MessageBase):
    """Схема для создания сообщения"""
    session_id: int = Field(..., description="ID сессии чата")
    parent_message_id: Optional[int] = Field(None, description="ID родительского сообщения")


class MessageUpdate(BaseModel):
    """Схема для обновления сообщения"""
    content: Optional[str] = Field(None, min_length=1)
    metadata: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_helpful: Optional[bool] = None
    user_rating: Optional[int] = Field(None, ge=1, le=5)


class MessageInDBBase(MessageBase):
    """Базовая схема сообщения в БД"""
    id: int
    session_id: int
    user_id: int
    parent_message_id: Optional[int] = None
    is_helpful: Optional[bool] = None
    user_rating: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Message(MessageInDBBase):
    """Схема сообщения для API ответов"""
    pass


class MessageInDB(MessageInDBBase):
    """Схема сообщения в базе данных"""
    pass


class MessageWithReplies(Message):
    """Схема сообщения с ответами"""
    replies: List[Message] = []


class ChatSessionBase(BaseModel):
    """Базовая схема сессии чата"""
    title: Optional[str] = Field(None, max_length=200, description="Название сессии")
    status: ChatSessionStatus = Field(ChatSessionStatus.active, description="Статус сессии")
    context: Optional[Dict[str, Any]] = Field(None, description="Контекст сессии")
    course_id: Optional[int] = Field(None, description="ID связанного курса")
    lesson_id: Optional[int] = Field(None, description="ID связанного урока")
    language: str = Field("ru", description="Язык общения")
    ai_model: str = Field("gpt-3.5-turbo", description="Используемая AI модель")
    max_messages: int = Field(100, ge=1, le=1000, description="Максимальное количество сообщений")
    auto_archive_after: int = Field(7, ge=1, description="Автоархивация через N дней")


class ChatSessionCreate(ChatSessionBase):
    """Схема для создания сессии чата"""
    pass


class ChatSessionUpdate(BaseModel):
    """Схема для обновления сессии чата"""
    title: Optional[str] = Field(None, max_length=200)
    status: Optional[ChatSessionStatus] = None
    context: Optional[Dict[str, Any]] = None
    course_id: Optional[int] = None
    lesson_id: Optional[int] = None
    language: Optional[str] = None
    ai_model: Optional[str] = None


class ChatSessionInDBBase(ChatSessionBase):
    """Базовая схема сессии чата в БД"""
    id: int
    user_id: int
    message_count: int = 0
    last_message_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatSession(ChatSessionInDBBase):
    """Схема сессии чата для API ответов"""
    pass


class ChatSessionInDB(ChatSessionInDBBase):
    """Схема сессии чата в базе данных"""
    pass


class ChatSessionWithMessages(ChatSession):
    """Схема сессии чата с сообщениями"""
    messages: List[Message] = []
    recent_messages: List[Message] = []


class ChatRequest(BaseModel):
    """Запрос к AI чату"""
    message: str = Field(..., min_length=1, max_length=4000, description="Сообщение пользователя")
    session_id: Optional[int] = Field(None, description="ID существующей сессии")
    context: Optional[Dict[str, Any]] = Field(None, description="Дополнительный контекст")
    course_id: Optional[int] = Field(None, description="ID курса для контекста")
    lesson_id: Optional[int] = Field(None, description="ID урока для контекста")
    attachments: Optional[List[str]] = Field(None, description="Прикрепленные файлы")
    gesture_data: Optional[Dict[str, Any]] = Field(None, description="Данные жестов")
    language: str = Field("ru", description="Предпочитаемый язык ответа")


class ChatResponse(BaseModel):
    """Ответ от AI чата"""
    message: str = Field(..., description="Ответ AI")
    session_id: int = Field(..., description="ID сессии")
    message_id: int = Field(..., description="ID сообщения")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Уровень уверенности")
    processing_time: float = Field(..., ge=0.0, description="Время обработки")
    suggested_actions: Optional[List[str]] = Field(None, description="Предлагаемые действия")
    related_content: Optional[List[Dict[str, Any]]] = Field(None, description="Связанный контент")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Дополнительные данные")


class GestureRecognitionRequest(BaseModel):
    """Запрос на распознавание жестов"""
    video_data: str = Field(..., description="Видео данные в base64")
    session_id: Optional[int] = Field(None, description="ID сессии")
    context: Optional[Dict[str, Any]] = Field(None, description="Контекст для распознавания")
    language: str = Field("ru", description="Язык для интерпретации")


class GestureRecognitionResponse(BaseModel):
    """Ответ распознавания жестов"""
    recognized_text: str = Field(..., description="Распознанный текст")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Уровень уверенности")
    gesture_sequence: List[Dict[str, Any]] = Field(..., description="Последовательность жестов")
    processing_time: float = Field(..., ge=0.0, description="Время обработки")
    suggestions: Optional[List[str]] = Field(None, description="Предложения по улучшению")


class ChatFilter(BaseModel):
    """Схема для фильтрации чатов"""
    user_id: Optional[int] = None
    status: Optional[ChatSessionStatus] = None
    course_id: Optional[int] = None
    lesson_id: Optional[int] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    search: Optional[str] = Field(None, description="Поиск по содержимому")
    language: Optional[str] = None


class ChatListResponse(BaseModel):
    """Схема ответа для списка чатов"""
    sessions: List[ChatSession]
    total: int
    page: int
    page_size: int
    total_pages: int


class MessageListResponse(BaseModel):
    """Схема ответа для списка сообщений"""
    messages: List[Message]
    total: int
    page: int
    page_size: int
    total_pages: int


class ChatStats(BaseModel):
    """Статистика чатов"""
    total_sessions: int
    active_sessions: int
    total_messages: int
    average_session_length: float
    most_common_topics: List[Dict[str, Any]]
    user_satisfaction: float
    response_time_avg: float
    gesture_recognition_accuracy: float


class AIModelConfig(BaseModel):
    """Конфигурация AI модели"""
    model_name: str
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1000, ge=1, le=4000)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    system_prompt: Optional[str] = None
    context_window: int = Field(10, ge=1, le=50)


class FeedbackCreate(BaseModel):
    """Схема для создания обратной связи"""
    message_id: int = Field(..., description="ID сообщения")
    is_helpful: bool = Field(..., description="Полезен ли ответ")
    rating: int = Field(..., ge=1, le=5, description="Оценка от 1 до 5")
    comment: Optional[str] = Field(None, max_length=1000, description="Комментарий")
    improvement_suggestions: Optional[str] = Field(None, max_length=1000, description="Предложения по улучшению")


class Feedback(BaseModel):
    """Схема обратной связи"""
    id: int
    message_id: int
    user_id: int
    is_helpful: bool
    rating: int
    comment: Optional[str] = None
    improvement_suggestions: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True