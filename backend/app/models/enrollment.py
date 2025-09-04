from sqlalchemy import Column, Integer, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.orm import relationship
from datetime import datetime

from app.core.database import Base


class Enrollment(Base):
    """Модель записи пользователя на курс"""
    __tablename__ = "enrollments"

    id = Column(INTEGER(unsigned=True), primary_key=True, index=True)
    user_id = Column(INTEGER(unsigned=True), ForeignKey("users.id"), nullable=False)
    course_id = Column(INTEGER(unsigned=True), ForeignKey("courses.id"), nullable=False)
    
    # Даты
    enrolled_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Статус
    is_active = Column(Boolean, default=True, nullable=False)
    is_completed = Column(Boolean, default=False, nullable=False)
    
    # Прогресс
    progress_percentage = Column(Integer, default=0, nullable=False)  # 0-100
    last_accessed_at = Column(DateTime, nullable=True)
    
    # Дополнительная информация
    notes = Column(Text, nullable=True)  # Заметки студента
    
    # Связи
    user = relationship("User", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")
    
    def __repr__(self):
        return f"<Enrollment(user_id={self.user_id}, course_id={self.course_id}, progress={self.progress_percentage}%)>"
    
    @property
    def is_in_progress(self) -> bool:
        """Проверить, находится ли курс в процессе изучения"""
        return self.is_active and not self.is_completed and self.progress_percentage > 0
    
    @property
    def duration_days(self) -> int:
        """Количество дней с момента записи"""
        if not self.enrolled_at:
            return 0
        return (datetime.utcnow() - self.enrolled_at).days
    
    def update_progress(self, percentage: int):
        """Обновить прогресс изучения"""
        self.progress_percentage = max(0, min(100, percentage))
        self.last_accessed_at = datetime.utcnow()
        
        if self.progress_percentage >= 100:
            self.is_completed = True
            self.completed_at = datetime.utcnow()
    
    def mark_completed(self):
        """Отметить курс как завершенный"""
        self.is_completed = True
        self.completed_at = datetime.utcnow()
        self.progress_percentage = 100
        self.last_accessed_at = datetime.utcnow()
    
    def deactivate(self):
        """Деактивировать запись (отписаться от курса)"""
        self.is_active = False