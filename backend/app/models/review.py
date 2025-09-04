from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base


class Review(Base):
    """Модель отзыва о курсе"""
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Связи с курсом и пользователем
    course_id = Column(
        Integer, 
        ForeignKey("courses.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="ID курса"
    )
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="ID автора отзыва"
    )
    
    # Содержание отзыва
    rating = Column(
        Integer, 
        nullable=False, 
        comment="Оценка от 1 до 5"
    )
    title = Column(String(200), nullable=True, comment="Заголовок отзыва")
    comment = Column(Text, nullable=True, comment="Текст отзыва")
    
    # Детализированные оценки
    content_rating = Column(Integer, nullable=True, comment="Оценка содержания (1-5)")
    instructor_rating = Column(Integer, nullable=True, comment="Оценка преподавателя (1-5)")
    difficulty_rating = Column(Integer, nullable=True, comment="Оценка сложности (1-5)")
    
    # Рекомендации
    would_recommend = Column(Boolean, nullable=True, comment="Рекомендует ли курс")
    
    # Статус и модерация
    is_approved = Column(Boolean, default=True, nullable=False, comment="Одобрен ли отзыв")
    is_featured = Column(Boolean, default=False, nullable=False, comment="Рекомендуемый отзыв")
    is_verified_purchase = Column(Boolean, default=False, nullable=False, comment="Подтвержденная покупка")
    
    # Полезность отзыва
    helpful_count = Column(Integer, default=0, nullable=False, comment="Количество 'полезно'")
    not_helpful_count = Column(Integer, default=0, nullable=False, comment="Количество 'не полезно'")
    
    # Временные метки
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False,
        comment="Дата создания отзыва"
    )
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now(),
        nullable=False,
        comment="Дата последнего обновления"
    )
    approved_at = Column(
        DateTime(timezone=True), 
        nullable=True,
        comment="Дата одобрения"
    )
    
    # Связи с другими таблицами
    course = relationship("Course", back_populates="reviews")
    user = relationship("User", back_populates="reviews")
    
    def __repr__(self):
        return f"<Review(id={self.id}, course_id={self.course_id}, user_id={self.user_id}, rating={self.rating})>"
    
    @property
    def is_positive(self) -> bool:
        """Положительный ли отзыв (4-5 звезд)"""
        return self.rating >= 4
    
    @property
    def is_negative(self) -> bool:
        """Отрицательный ли отзыв (1-2 звезды)"""
        return self.rating <= 2
    
    @property
    def is_neutral(self) -> bool:
        """Нейтральный ли отзыв (3 звезды)"""
        return self.rating == 3
    
    @property
    def helpfulness_ratio(self) -> float:
        """Соотношение полезности отзыва"""
        total_votes = self.helpful_count + self.not_helpful_count
        if total_votes == 0:
            return 0.0
        return round(self.helpful_count / total_votes, 2)
    
    @property
    def average_detailed_rating(self) -> float:
        """Средняя детализированная оценка"""
        ratings = []
        if self.content_rating:
            ratings.append(self.content_rating)
        if self.instructor_rating:
            ratings.append(self.instructor_rating)
        if self.difficulty_rating:
            ratings.append(self.difficulty_rating)
        
        if not ratings:
            return self.rating
        
        return round(sum(ratings) / len(ratings), 1)
    
    @property
    def stars_display(self) -> str:
        """Отображение звезд"""
        full_stars = "★" * self.rating
        empty_stars = "☆" * (5 - self.rating)
        return full_stars + empty_stars
    
    def mark_helpful(self, is_helpful: bool = True):
        """Отметить отзыв как полезный или нет"""
        if is_helpful:
            self.helpful_count += 1
        else:
            self.not_helpful_count += 1
    
    def approve(self):
        """Одобрить отзыв"""
        self.is_approved = True
        self.approved_at = func.now()
    
    def feature(self):
        """Сделать отзыв рекомендуемым"""
        self.is_featured = True
    
    def verify_purchase(self):
        """Подтвердить покупку"""
        self.is_verified_purchase = True