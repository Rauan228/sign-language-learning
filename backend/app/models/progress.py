from sqlalchemy import Column, Integer, String, Enum, DateTime, ForeignKey, Float, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class ProgressStatusEnum(str, enum.Enum):
    """Статус прогресса"""
    NOT_STARTED = "not_started"  # Не начат
    IN_PROGRESS = "in_progress"  # В процессе
    COMPLETED = "completed"  # Завершен
    FAILED = "failed"  # Провален


class Progress(Base):
    """Модель прогресса студента"""
    __tablename__ = "progress"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Связи с пользователем, курсом и уроком
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="ID студента"
    )
    course_id = Column(
        Integer, 
        ForeignKey("courses.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="ID курса"
    )
    lesson_id = Column(
        Integer, 
        ForeignKey("lessons.id", ondelete="CASCADE"), 
        nullable=True,
        index=True,
        comment="ID урока (может быть NULL для общего прогресса по курсу)"
    )
    
    # Статус и результаты
    status = Column(
        Enum(ProgressStatusEnum), 
        default=ProgressStatusEnum.NOT_STARTED, 
        nullable=False,
        comment="Статус прохождения"
    )
    score = Column(Integer, nullable=True, comment="Баллы (из тестов)")
    max_score = Column(Integer, nullable=True, comment="Максимальные баллы")
    percentage = Column(Float, default=0.0, nullable=False, comment="Процент выполнения")
    
    # Попытки и время
    attempts = Column(Integer, default=0, nullable=False, comment="Количество попыток")
    time_spent = Column(Integer, default=0, nullable=False, comment="Время, потраченное в секундах")
    
    # Дополнительные данные (JSON)
    extra_data = Column(
        JSON, 
        nullable=True, 
        comment="Дополнительные данные о прогрессе"
    )
    
    # Временные метки
    started_at = Column(
        DateTime(timezone=True), 
        nullable=True,
        comment="Время начала"
    )
    completed_at = Column(
        DateTime(timezone=True), 
        nullable=True,
        comment="Время завершения"
    )
    last_accessed_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Последний доступ"
    )
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False,
        comment="Дата создания записи"
    )
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now(),
        nullable=False,
        comment="Дата последнего обновления"
    )
    
    # Связи с другими таблицами
    user = relationship("User", back_populates="progress")
    course = relationship("Course", back_populates="progress")
    lesson = relationship("Lesson", back_populates="progress")
    
    def __repr__(self):
        return f"<Progress(id={self.id}, user_id={self.user_id}, course_id={self.course_id}, lesson_id={self.lesson_id}, status='{self.status}')>"
    
    @property
    def is_completed(self) -> bool:
        """Проверка, завершен ли прогресс"""
        return self.status == ProgressStatusEnum.COMPLETED
    
    @property
    def is_in_progress(self) -> bool:
        """Проверка, в процессе ли прогресс"""
        return self.status == ProgressStatusEnum.IN_PROGRESS
    
    @property
    def is_failed(self) -> bool:
        """Проверка, провален ли прогресс"""
        return self.status == ProgressStatusEnum.FAILED
    
    @property
    def score_percentage(self) -> float:
        """Процент набранных баллов"""
        if not self.max_score or self.max_score == 0:
            return 0.0
        return round((self.score or 0) / self.max_score * 100, 1)
    
    @property
    def time_spent_formatted(self) -> str:
        """Форматированное время, потраченное на изучение"""
        if self.time_spent < 60:
            return f"{self.time_spent} сек"
        elif self.time_spent < 3600:
            minutes = self.time_spent // 60
            seconds = self.time_spent % 60
            return f"{minutes} мин {seconds} сек"
        else:
            hours = self.time_spent // 3600
            minutes = (self.time_spent % 3600) // 60
            return f"{hours} ч {minutes} мин"
    
    def start_progress(self):
        """Начать прогресс"""
        if self.status == ProgressStatusEnum.NOT_STARTED:
            self.status = ProgressStatusEnum.IN_PROGRESS
            self.started_at = func.now()
    
    def complete_progress(self, score: int = None, max_score: int = None):
        """Завершить прогресс"""
        self.status = ProgressStatusEnum.COMPLETED
        self.completed_at = func.now()
        self.percentage = 100.0
        
        if score is not None:
            self.score = score
        if max_score is not None:
            self.max_score = max_score
    
    def fail_progress(self):
        """Отметить прогресс как проваленный"""
        self.status = ProgressStatusEnum.FAILED
        self.completed_at = func.now()