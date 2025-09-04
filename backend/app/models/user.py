from sqlalchemy import Column, Integer, String, Enum, DateTime, Boolean
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class SignLanguageEnum(str, enum.Enum):
    """Перечисление языков жестов"""
    RU = "ru"  # Русский жестовый язык
    KZ = "kz"  # Казахский жестовый язык
    ASL = "asl"  # American Sign Language
    BSL = "bsl"  # British Sign Language


class UserRoleEnum(str, enum.Enum):
    """Перечисление ролей пользователей"""
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"


class User(Base):
    """Модель пользователя"""
    __tablename__ = "users"
    
    id = Column(INTEGER(unsigned=True), primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False, comment="Имя пользователя")
    email = Column(String(150), unique=True, index=True, nullable=False, comment="Email")
    phone = Column(String(20), nullable=True, comment="Телефон")
    password_hash = Column(String(255), nullable=False, comment="Хэш пароля")
    sign_language = Column(
        Enum(SignLanguageEnum), 
        default=SignLanguageEnum.RU, 
        nullable=False,
        comment="Предпочитаемый язык жестов"
    )
    role = Column(
        Enum(UserRoleEnum), 
        default=UserRoleEnum.STUDENT, 
        nullable=False,
        comment="Роль пользователя"
    )
    is_active = Column(Boolean, default=True, nullable=False, comment="Активен ли пользователь")
    is_verified = Column(Boolean, default=False, nullable=False, comment="Подтвержден ли email")
    avatar_url = Column(String(255), nullable=True, comment="URL аватара")
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False,
        comment="Дата регистрации"
    )
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now(),
        nullable=False,
        comment="Дата последнего обновления"
    )
    
    # Связи с другими таблицами
    progress = relationship("Progress", back_populates="user", cascade="all, delete-orphan")
    ai_chats = relationship("AIChat", back_populates="user", cascade="all, delete-orphan")
    reviews = relationship("Review", back_populates="user", cascade="all, delete-orphan")
    enrollments = relationship("Enrollment", back_populates="user", cascade="all, delete-orphan")
    created_courses = relationship(
        "Course", 
        foreign_keys="Course.teacher_id",
        back_populates="teacher",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"
    
    @property
    def is_student(self) -> bool:
        """Проверка, является ли пользователь студентом"""
        return self.role == UserRoleEnum.STUDENT
    
    @property
    def is_teacher(self) -> bool:
        """Проверка, является ли пользователь преподавателем"""
        return self.role == UserRoleEnum.TEACHER
    
    @property
    def is_admin(self) -> bool:
        """Проверка, является ли пользователь администратором"""
        return self.role == UserRoleEnum.ADMIN
    
    @property
    def full_name(self) -> str:
        """Полное имя пользователя"""
        return self.name
    
    @property
    def rating(self) -> float:
        """Рейтинг преподавателя (заглушка)"""
        # TODO: Реализовать расчет рейтинга на основе отзывов
        return 0.0