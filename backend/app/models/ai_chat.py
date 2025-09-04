from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class MessageTypeEnum(str, enum.Enum):
    """Тип сообщения в чате"""
    TEXT = "text"  # Текстовое сообщение
    GESTURE = "gesture"  # Жест (видео/изображение)
    VOICE = "voice"  # Голосовое сообщение
    IMAGE = "image"  # Изображение


class ResponseTypeEnum(str, enum.Enum):
    """Тип ответа AI"""
    TEXT = "text"  # Текстовый ответ
    GESTURE_3D = "gesture_3d"  # 3D анимация жеста
    VIDEO = "video"  # Видео ответ
    MIXED = "mixed"  # Смешанный ответ (текст + жест)


class AIChat(Base):
    """Модель чата с AI ассистентом"""
    __tablename__ = "ai_chat"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Связь с пользователем
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="ID пользователя"
    )
    
    # Сессия чата (для группировки сообщений)
    session_id = Column(String(100), nullable=True, index=True, comment="ID сессии чата")
    
    # Вопрос пользователя
    question = Column(Text, nullable=False, comment="Вопрос пользователя")
    question_type = Column(
        Enum(MessageTypeEnum), 
        default=MessageTypeEnum.TEXT, 
        nullable=False,
        comment="Тип вопроса"
    )
    question_file_url = Column(String(255), nullable=True, comment="URL файла вопроса (если есть)")
    
    # Ответ AI
    answer = Column(Text, nullable=False, comment="Ответ AI ассистента")
    answer_type = Column(
        Enum(ResponseTypeEnum), 
        default=ResponseTypeEnum.TEXT, 
        nullable=False,
        comment="Тип ответа"
    )
    answer_file_url = Column(String(255), nullable=True, comment="URL файла ответа (видео/3D модель)")
    
    # Дополнительные данные
    context_data = Column(
        JSON, 
        nullable=True, 
        comment="Контекстные данные для AI (курс, урок и т.д.)"
    )
    
    # Метаданные
    confidence_score = Column(Integer, nullable=True, comment="Уверенность AI в ответе (0-100)")
    processing_time = Column(Integer, nullable=True, comment="Время обработки запроса в мс")
    
    # Обратная связь от пользователя
    user_rating = Column(Integer, nullable=True, comment="Оценка ответа пользователем (1-5)")
    user_feedback = Column(Text, nullable=True, comment="Отзыв пользователя об ответе")
    
    # Флаги
    is_helpful = Column(String(10), nullable=True, comment="Был ли ответ полезен (yes/no/null)")
    is_flagged = Column(String(10), default="no", nullable=False, comment="Отмечен ли как проблемный")
    
    # Временные метки
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False,
        comment="Время создания сообщения"
    )
    
    # Связи с другими таблицами
    user = relationship("User", back_populates="ai_chats")
    
    def __repr__(self):
        return f"<AIChat(id={self.id}, user_id={self.user_id}, question='{self.question[:50]}...')>"
    
    @property
    def has_question_file(self) -> bool:
        """Есть ли файл в вопросе"""
        return bool(self.question_file_url)
    
    @property
    def has_answer_file(self) -> bool:
        """Есть ли файл в ответе"""
        return bool(self.answer_file_url)
    
    @property
    def is_gesture_question(self) -> bool:
        """Является ли вопрос жестом"""
        return self.question_type == MessageTypeEnum.GESTURE
    
    @property
    def is_3d_answer(self) -> bool:
        """Является ли ответ 3D анимацией"""
        return self.answer_type == ResponseTypeEnum.GESTURE_3D
    
    @property
    def has_user_feedback(self) -> bool:
        """Есть ли обратная связь от пользователя"""
        return bool(self.user_rating or self.user_feedback or self.is_helpful)
    
    def set_helpful(self, is_helpful: bool):
        """Установить, был ли ответ полезен"""
        self.is_helpful = "yes" if is_helpful else "no"
    
    def flag_as_problematic(self, reason: str = None):
        """Отметить как проблемный"""
        self.is_flagged = "yes"
        if reason and self.context_data:
            self.context_data["flag_reason"] = reason
        elif reason:
            self.context_data = {"flag_reason": reason}
    
    def add_context(self, course_id: int = None, lesson_id: int = None, **kwargs):
        """Добавить контекстную информацию"""
        context = self.context_data or {}
        
        if course_id:
            context["course_id"] = course_id
        if lesson_id:
            context["lesson_id"] = lesson_id
        
        context.update(kwargs)
        self.context_data = context