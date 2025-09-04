from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ReviewType(str, Enum):
    """Типы отзывов"""
    course = "course"
    lesson = "lesson"
    teacher = "teacher"
    platform = "platform"


class ReviewStatus(str, Enum):
    """Статусы отзывов"""
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    hidden = "hidden"


class ReviewBase(BaseModel):
    """Базовая схема отзыва"""
    rating: int = Field(..., ge=1, le=5, description="Оценка от 1 до 5")
    title: Optional[str] = Field(None, max_length=200, description="Заголовок отзыва")
    content: str = Field(..., min_length=10, max_length=2000, description="Содержание отзыва")
    pros: Optional[str] = Field(None, max_length=1000, description="Плюсы")
    cons: Optional[str] = Field(None, max_length=1000, description="Минусы")
    review_type: ReviewType = Field(..., description="Тип отзыва")
    is_anonymous: bool = Field(False, description="Анонимный ли отзыв")
    would_recommend: bool = Field(True, description="Рекомендует ли пользователь")
    difficulty_rating: Optional[int] = Field(None, ge=1, le=5, description="Оценка сложности")
    quality_rating: Optional[int] = Field(None, ge=1, le=5, description="Оценка качества")
    support_rating: Optional[int] = Field(None, ge=1, le=5, description="Оценка поддержки")
    tags: Optional[List[str]] = Field(None, description="Теги отзыва")

    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Содержание отзыва должно содержать минимум 10 символов')
        return v.strip()

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        if v and len(v) > 10:
            raise ValueError('Максимальное количество тегов: 10')
        return v


class ReviewCreate(ReviewBase):
    """Схема для создания отзыва"""
    course_id: Optional[int] = Field(None, description="ID курса")
    lesson_id: Optional[int] = Field(None, description="ID урока")
    teacher_id: Optional[int] = Field(None, description="ID преподавателя")

    @field_validator('course_id', 'lesson_id', 'teacher_id')
    @classmethod
    def validate_target(cls, v, info):
        # В Pydantic V2 нужно использовать model_validate для доступа к другим полям
        # Пока упростим валидацию
        return v


class ReviewUpdate(BaseModel):
    """Схема для обновления отзыва"""
    rating: Optional[int] = Field(None, ge=1, le=5)
    title: Optional[str] = Field(None, max_length=200)
    content: Optional[str] = Field(None, min_length=10, max_length=2000)
    pros: Optional[str] = Field(None, max_length=1000)
    cons: Optional[str] = Field(None, max_length=1000)
    is_anonymous: Optional[bool] = None
    would_recommend: Optional[bool] = None
    difficulty_rating: Optional[int] = Field(None, ge=1, le=5)
    quality_rating: Optional[int] = Field(None, ge=1, le=5)
    support_rating: Optional[int] = Field(None, ge=1, le=5)
    tags: Optional[List[str]] = None


class ReviewInDBBase(ReviewBase):
    """Базовая схема отзыва в БД"""
    id: int
    user_id: int
    course_id: Optional[int] = None
    lesson_id: Optional[int] = None
    teacher_id: Optional[int] = None
    status: ReviewStatus = ReviewStatus.pending
    helpful_count: int = 0
    not_helpful_count: int = 0
    reply_count: int = 0
    is_verified_purchase: bool = False
    moderation_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Review(ReviewInDBBase):
    """Схема отзыва для API ответов"""
    pass


class ReviewInDB(ReviewInDBBase):
    """Схема отзыва в базе данных"""
    pass


class ReviewWithAuthor(Review):
    """Схема отзыва с информацией об авторе"""
    author_name: Optional[str] = None
    author_avatar: Optional[str] = None
    author_level: Optional[str] = None
    author_courses_completed: int = 0


class ReviewWithReplies(ReviewWithAuthor):
    """Схема отзыва с ответами"""
    replies: List['ReviewReply'] = []


class ReviewReplyBase(BaseModel):
    """Базовая схема ответа на отзыв"""
    content: str = Field(..., min_length=5, max_length=1000, description="Содержание ответа")
    is_official: bool = Field(False, description="Официальный ли ответ")


class ReviewReplyCreate(ReviewReplyBase):
    """Схема для создания ответа на отзыв"""
    review_id: int = Field(..., description="ID отзыва")


class ReviewReplyUpdate(BaseModel):
    """Схема для обновления ответа на отзыв"""
    content: Optional[str] = Field(None, min_length=5, max_length=1000)
    is_official: Optional[bool] = None


class ReviewReplyInDBBase(ReviewReplyBase):
    """Базовая схема ответа на отзыв в БД"""
    id: int
    review_id: int
    user_id: int
    helpful_count: int = 0
    not_helpful_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ReviewReply(ReviewReplyInDBBase):
    """Схема ответа на отзыв для API ответов"""
    pass


class ReviewReplyInDB(ReviewReplyInDBBase):
    """Схема ответа на отзыв в базе данных"""
    pass


class ReviewReplyWithAuthor(ReviewReply):
    """Схема ответа на отзыв с информацией об авторе"""
    author_name: str
    author_avatar: Optional[str] = None
    author_role: str


class ReviewHelpfulnessBase(BaseModel):
    """Базовая схема оценки полезности отзыва"""
    is_helpful: bool = Field(..., description="Полезен ли отзыв")


class ReviewHelpfulnessCreate(ReviewHelpfulnessBase):
    """Схема для создания оценки полезности отзыва"""
    review_id: int = Field(..., description="ID отзыва")


class ReviewHelpfulness(ReviewHelpfulnessBase):
    """Схема оценки полезности отзыва"""
    id: int
    review_id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ReviewFilter(BaseModel):
    """Схема для фильтрации отзывов"""
    course_id: Optional[int] = None
    lesson_id: Optional[int] = None
    teacher_id: Optional[int] = None
    user_id: Optional[int] = None
    review_type: Optional[ReviewType] = None
    status: Optional[ReviewStatus] = None
    min_rating: Optional[int] = Field(None, ge=1, le=5)
    max_rating: Optional[int] = Field(None, ge=1, le=5)
    would_recommend: Optional[bool] = None
    is_verified_purchase: Optional[bool] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    search: Optional[str] = Field(None, description="Поиск по содержимому")
    tags: Optional[List[str]] = None
    sort_by: Optional[str] = Field("created_at", description="Поле для сортировки")
    sort_order: Optional[str] = Field("desc", description="Порядок сортировки (asc/desc)")


class ReviewListResponse(BaseModel):
    """Схема ответа для списка отзывов"""
    reviews: List[ReviewWithAuthor]
    total: int
    page: int
    page_size: int
    total_pages: int
    average_rating: float
    rating_distribution: Dict[int, int]


class ReviewStats(BaseModel):
    """Статистика отзывов"""
    total_reviews: int
    average_rating: float
    rating_distribution: Dict[int, int]
    recommendation_rate: float
    verified_reviews_count: int
    recent_reviews_count: int
    response_rate: float
    average_response_time: float
    most_common_tags: List[Dict[str, Any]]
    sentiment_analysis: Dict[str, float]


class CourseReviewSummary(BaseModel):
    """Сводка отзывов о курсе"""
    course_id: int
    total_reviews: int
    average_rating: float
    rating_distribution: Dict[int, int]
    recommendation_rate: float
    difficulty_average: float
    quality_average: float
    support_average: float
    recent_reviews: List[ReviewWithAuthor]
    top_pros: List[str]
    top_cons: List[str]


class TeacherReviewSummary(BaseModel):
    """Сводка отзывов о преподавателе"""
    teacher_id: int
    total_reviews: int
    average_rating: float
    rating_distribution: Dict[int, int]
    courses_reviewed: int
    teaching_quality: float
    communication_rating: float
    helpfulness_rating: float
    recent_reviews: List[ReviewWithAuthor]


class ReviewModerationAction(BaseModel):
    """Действие модерации отзыва"""
    review_id: int
    action: str = Field(..., description="Действие (approve/reject/hide)")
    reason: Optional[str] = Field(None, description="Причина действия")
    notes: Optional[str] = Field(None, description="Заметки модератора")


class ReviewReport(BaseModel):
    """Жалоба на отзыв"""
    id: int
    review_id: int
    reporter_id: int
    reason: str
    description: Optional[str] = None
    status: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    moderator_id: Optional[int] = None
    moderator_notes: Optional[str] = None

    class Config:
        from_attributes = True


class ReviewReportCreate(BaseModel):
    """Схема для создания жалобы на отзыв"""
    review_id: int = Field(..., description="ID отзыва")
    reason: str = Field(..., description="Причина жалобы")
    description: Optional[str] = Field(None, max_length=500, description="Описание проблемы")


# Обновляем forward reference
ReviewWithReplies.model_rebuild()