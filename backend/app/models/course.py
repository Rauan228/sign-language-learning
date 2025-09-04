from sqlalchemy import Column, Integer, String, Text, Enum, DECIMAL, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
from .user import SignLanguageEnum
import enum


class CourseLevelEnum(str, enum.Enum):
    """Перечисление уровней курсов"""
    BEGINNER = "beginner"  # Начинающий
    BASIC = "basic"  # Базовый
    INTERMEDIATE = "intermediate"  # Средний
    ADVANCED = "advanced"  # Продвинутый
    EXPERT = "expert"  # Эксперт


class CourseStatusEnum(str, enum.Enum):
    """Статус курса"""
    DRAFT = "draft"  # Черновик
    PUBLISHED = "published"  # Опубликован
    ARCHIVED = "archived"  # Архивирован


class Course(Base):
    """Модель курса"""
    __tablename__ = "courses"
    
    id = Column(INTEGER(unsigned=True), primary_key=True, index=True, autoincrement=True)
    title = Column(String(200), nullable=False, index=True, comment="Название курса")
    description = Column(Text, nullable=False, comment="Описание курса")
    short_description = Column(String(500), nullable=True, comment="Краткое описание")
    subject = Column(String(100), nullable=False, index=True, comment="Предмет")
    category = Column(String(100), nullable=False, index=True, comment="Категория")
    level = Column(
        Enum(CourseLevelEnum), 
        nullable=False, 
        index=True,
        comment="Уровень сложности"
    )
    sign_language = Column(
        Enum(SignLanguageEnum), 
        nullable=False, 
        index=True,
        comment="Язык жестов"
    )
    price = Column(DECIMAL(10, 2), nullable=False, default=0.00, comment="Цена курса")
    original_price = Column(DECIMAL(10, 2), nullable=True, comment="Первоначальная цена (для скидок)")
    is_free = Column(Boolean, default=False, nullable=False, comment="Бесплатный ли курс")
    
    # Рейтинг и статистика
    rating = Column(Float, default=0.0, nullable=False, comment="Средний рейтинг")
    reviews_count = Column(Integer, default=0, nullable=False, comment="Количество отзывов")
    students_count = Column(Integer, default=0, nullable=False, comment="Количество студентов")
    
    # Метаданные
    duration_hours = Column(Integer, nullable=True, comment="Продолжительность в часах")
    thumbnail_url = Column(String(255), nullable=True, comment="URL миниатюры")
    trailer_url = Column(String(255), nullable=True, comment="URL трейлера")
    
    # Статус и флаги
    status = Column(
        Enum(CourseStatusEnum), 
        default=CourseStatusEnum.DRAFT, 
        nullable=False,
        comment="Статус курса"
    )
    is_featured = Column(Boolean, default=False, nullable=False, comment="Рекомендуемый курс")
    is_new = Column(Boolean, default=True, nullable=False, comment="Новый курс")
    is_bestseller = Column(Boolean, default=False, nullable=False, comment="Бестселлер")
    
    # Преподаватель
    teacher_id = Column(
        INTEGER(unsigned=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        comment="ID преподавателя"
    )
    
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
    published_at = Column(
        DateTime(timezone=True), 
        nullable=True,
        comment="Дата публикации"
    )
    
    # Связи с другими таблицами
    teacher = relationship("User", foreign_keys=[teacher_id], back_populates="created_courses")
    lessons = relationship("Lesson", back_populates="course", cascade="all, delete-orphan", order_by="Lesson.order_num")
    progress = relationship("Progress", back_populates="course", cascade="all, delete-orphan")
    reviews = relationship("Review", back_populates="course", cascade="all, delete-orphan")
    enrollments = relationship("Enrollment", back_populates="course", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Course(id={self.id}, title='{self.title}', level='{self.level}')>"
    
    @property
    def is_published(self) -> bool:
        """Проверка, опубликован ли курс"""
        return self.status == CourseStatusEnum.PUBLISHED
    
    @property
    def lessons_count(self) -> int:
        """Количество уроков в курсе"""
        return len(self.lessons) if self.lessons else 0
    
    def update_rating(self):
        """Обновление рейтинга курса на основе отзывов"""
        if self.reviews:
            total_rating = sum(review.rating for review in self.reviews)
            self.rating = round(total_rating / len(self.reviews), 1)
            self.reviews_count = len(self.reviews)
        else:
            self.rating = 0.0
            self.reviews_count = 0