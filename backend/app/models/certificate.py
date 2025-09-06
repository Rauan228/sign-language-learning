from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

from app.core.database import Base


class Certificate(Base):
    """Модель сертификата"""
    __tablename__ = "certificates"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    
    # Информация о сертификате
    certificate_number = Column(String(50), unique=True, nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Данные о прохождении
    completion_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    total_score = Column(Integer, default=0)
    max_possible_score = Column(Integer, default=0)
    percentage = Column(Integer, default=0)  # Процент прохождения
    
    # Время изучения
    total_time_spent = Column(Integer, default=0)  # В минутах
    
    # Статус сертификата
    is_active = Column(Boolean, default=True)
    
    # Метаданные
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Связи
    user = relationship("User", back_populates="certificates")
    course = relationship("Course", back_populates="certificates")
    
    def __repr__(self):
        return f"<Certificate(id={self.id}, number={self.certificate_number}, user_id={self.user_id}, course_id={self.course_id})>"