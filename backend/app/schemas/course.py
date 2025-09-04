from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class CourseLevel(str, Enum):
    """Уровни сложности курса"""
    beginner = "beginner"
    basic = "basic"
    intermediate = "intermediate"
    advanced = "advanced"
    expert = "expert"


class SignLanguage(str, Enum):
    """Поддерживаемые языки жестов"""
    ru = "ru"
    kz = "kz"
    asl = "asl"
    bsl = "bsl"


class CourseStatus(str, Enum):
    """Статусы курса"""
    draft = "draft"
    published = "published"
    archived = "archived"


class CourseBase(BaseModel):
    """Базовая схема курса"""
    title: str = Field(..., min_length=1, max_length=200, description="Название курса")
    description: str = Field(..., min_length=1, description="Описание курса")
    short_description: Optional[str] = Field(None, max_length=500, description="Краткое описание")
    subject: str = Field(..., min_length=1, max_length=100, description="Предмет")
    category: str = Field(..., min_length=1, max_length=100, description="Категория")
    level: CourseLevel = Field(..., description="Уровень сложности")
    sign_language: SignLanguage = Field(..., description="Язык жестов")
    price: float = Field(0.0, ge=0, description="Цена курса")
    original_price: Optional[float] = Field(None, ge=0, description="Первоначальная цена")
    is_free: bool = Field(False, description="Бесплатный ли курс")
    duration_hours: Optional[int] = Field(None, ge=0, description="Продолжительность в часах")
    thumbnail_url: Optional[str] = Field(None, description="URL миниатюры")
    trailer_url: Optional[str] = Field(None, description="URL трейлера")
    status: CourseStatus = Field(CourseStatus.draft, description="Статус курса")
    is_featured: bool = Field(False, description="Рекомендуемый курс")
    is_new: bool = Field(True, description="Новый курс")
    is_bestseller: bool = Field(False, description="Бестселлер")

    @validator('price')
    def validate_price(cls, v, values):
        if values.get('is_free') and v > 0:
            raise ValueError('Бесплатный курс не может иметь цену больше 0')
        return v

    @validator('original_price')
    def validate_original_price(cls, v, values):
        if v is not None and 'price' in values and v < values['price']:
            raise ValueError('Первоначальная цена не может быть меньше текущей цены')
        return v


class CourseCreate(CourseBase):
    """Схема для создания курса"""
    teacher_id: int = Field(..., description="ID преподавателя")


class CourseUpdate(BaseModel):
    """Схема для обновления курса"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=1)
    short_description: Optional[str] = Field(None, max_length=500)
    subject: Optional[str] = Field(None, min_length=1, max_length=100)
    category: Optional[str] = Field(None, min_length=1, max_length=100)
    level: Optional[CourseLevel] = None
    sign_language: Optional[SignLanguage] = None
    price: Optional[float] = Field(None, ge=0)
    original_price: Optional[float] = Field(None, ge=0)
    is_free: Optional[bool] = None
    duration_hours: Optional[int] = Field(None, ge=0)
    thumbnail_url: Optional[str] = None
    trailer_url: Optional[str] = None
    status: Optional[CourseStatus] = None
    is_featured: Optional[bool] = None
    is_new: Optional[bool] = None
    is_bestseller: Optional[bool] = None

    @validator('price')
    def validate_price(cls, v, values):
        if v is not None and values.get('is_free') and v > 0:
            raise ValueError('Бесплатный курс не может иметь цену больше 0')
        return v


class CourseInDBBase(CourseBase):
    """Базовая схема курса в БД"""
    id: int
    teacher_id: int
    rating: float = 0.0
    reviews_count: int = 0
    students_count: int = 0
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class Course(CourseInDBBase):
    """Схема курса для API ответов"""
    pass


class CourseInDB(CourseInDBBase):
    """Схема курса в базе данных"""
    pass


class CourseWithTeacher(Course):
    """Схема курса с информацией о преподавателе"""
    teacher_name: str
    teacher_email: str


class CourseListResponse(BaseModel):
    """Схема ответа для списка курсов"""
    courses: List[Course]
    total: int
    page: int
    page_size: int
    total_pages: int


class CourseFilter(BaseModel):
    """Схема для фильтрации курсов"""
    subject: Optional[str] = None
    category: Optional[str] = None
    level: Optional[CourseLevel] = None
    sign_language: Optional[SignLanguage] = None
    is_free: Optional[bool] = None
    is_featured: Optional[bool] = None
    is_new: Optional[bool] = None
    is_bestseller: Optional[bool] = None
    teacher_id: Optional[int] = None
    min_price: Optional[float] = Field(None, ge=0)
    max_price: Optional[float] = Field(None, ge=0)
    search: Optional[str] = Field(None, description="Поиск по названию и описанию")

    @validator('max_price')
    def validate_price_range(cls, v, values):
        if v is not None and 'min_price' in values and values['min_price'] is not None:
            if v < values['min_price']:
                raise ValueError('Максимальная цена не может быть меньше минимальной')
        return v


class CourseStats(BaseModel):
    """Статистика курса"""
    total_courses: int
    published_courses: int
    draft_courses: int
    archived_courses: int
    total_students: int
    average_rating: float
    total_revenue: float