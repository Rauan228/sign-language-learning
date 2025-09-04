from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class LessonTypeEnum(str, enum.Enum):
    """Тип урока"""
    VIDEO = "video"  # Видео урок
    THEORY = "theory"  # Теоретический материал
    PRACTICE = "practice"  # Практическое задание
    QUIZ = "quiz"  # Тест/квиз
    ASSIGNMENT = "assignment"  # Задание


class Lesson(Base):
    """Модель урока"""
    __tablename__ = "lessons"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    course_id = Column(
        Integer, 
        ForeignKey("courses.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="ID курса"
    )
    title = Column(String(200), nullable=False, comment="Название урока")
    description = Column(Text, nullable=True, comment="Описание урока")
    
    # Контент урока
    video_url = Column(String(255), nullable=True, comment="URL видео")
    video_duration = Column(Integer, nullable=True, comment="Длительность видео в секундах")
    
    # Материалы урока (JSON структура)
    materials = Column(
        JSON, 
        nullable=True, 
        comment="Материалы урока в формате JSON"
    )
    
    # Теоретический контент
    theory_content = Column(Text, nullable=True, comment="Теоретический материал")
    
    # Практические задания
    practice_tasks = Column(
        JSON, 
        nullable=True, 
        comment="Практические задания в формате JSON"
    )
    
    # Тест/квиз
    quiz_questions = Column(
        JSON, 
        nullable=True, 
        comment="Вопросы теста в формате JSON"
    )
    
    # Метаданные
    order_num = Column(Integer, nullable=False, comment="Порядковый номер урока в курсе")
    lesson_type = Column(String(50), default="video", nullable=False, comment="Тип урока")
    is_free = Column(Boolean, default=False, nullable=False, comment="Бесплатный ли урок")
    is_published = Column(Boolean, default=False, nullable=False, comment="Опубликован ли урок")
    
    # Требования для прохождения
    min_score = Column(Integer, default=0, nullable=False, comment="Минимальный балл для прохождения")
    max_attempts = Column(Integer, default=3, nullable=False, comment="Максимальное количество попыток")
    
    # Временные метки
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False,
        comment="Дата создания"
    )
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now(),
        nullable=False,
        comment="Дата последнего обновления"
    )
    
    # Связи с другими таблицами
    course = relationship("Course", back_populates="lessons")
    progress = relationship("Progress", back_populates="lesson", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Lesson(id={self.id}, title='{self.title}', course_id={self.course_id}, order={self.order_num})>"
    
    @property
    def duration_formatted(self) -> str:
        """Форматированная длительность видео"""
        if not self.video_duration:
            return "0:00"
        
        minutes = self.video_duration // 60
        seconds = self.video_duration % 60
        return f"{minutes}:{seconds:02d}"
    
    @property
    def has_video(self) -> bool:
        """Есть ли видео в уроке"""
        return bool(self.video_url)
    
    @property
    def has_quiz(self) -> bool:
        """Есть ли тест в уроке"""
        return bool(self.quiz_questions)
    
    @property
    def has_practice(self) -> bool:
        """Есть ли практические задания"""
        return bool(self.practice_tasks)
    
    def get_materials_list(self) -> list:
        """Получить список материалов урока"""
        if not self.materials:
            return []
        
        if isinstance(self.materials, list):
            return self.materials
        elif isinstance(self.materials, dict):
            return self.materials.get('files', [])
        
        return []
    
    def get_quiz_questions_count(self) -> int:
        """Количество вопросов в тесте"""
        if not self.quiz_questions:
            return 0
        
        if isinstance(self.quiz_questions, list):
            return len(self.quiz_questions)
        elif isinstance(self.quiz_questions, dict):
            return len(self.quiz_questions.get('questions', []))
        
        return 0