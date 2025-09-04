from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class LessonType(str, Enum):
    """Типы уроков"""
    video = "video"
    theory = "theory"
    practice = "practice"
    quiz = "quiz"
    mixed = "mixed"


class LessonBase(BaseModel):
    """Базовая схема урока"""
    title: str = Field(..., min_length=1, max_length=200, description="Название урока")
    description: Optional[str] = Field(None, description="Описание урока")
    video_url: Optional[str] = Field(None, description="URL видео")
    video_duration: Optional[int] = Field(None, ge=0, description="Длительность видео в секундах")
    materials: Optional[Dict[str, Any]] = Field(None, description="Материалы урока в формате JSON")
    theory_content: Optional[str] = Field(None, description="Теоретический материал")
    practice_tasks: Optional[Dict[str, Any]] = Field(None, description="Практические задания")
    quiz_questions: Optional[Dict[str, Any]] = Field(None, description="Вопросы теста")
    order_num: int = Field(..., ge=1, description="Порядковый номер урока в курсе")
    lesson_type: LessonType = Field(LessonType.video, description="Тип урока")
    is_free: bool = Field(False, description="Бесплатный ли урок")
    is_published: bool = Field(False, description="Опубликован ли урок")
    min_score: int = Field(0, ge=0, le=100, description="Минимальный балл для прохождения")
    max_attempts: int = Field(3, ge=1, description="Максимальное количество попыток")

    @validator('video_duration')
    def validate_video_duration(cls, v, values):
        if values.get('lesson_type') == LessonType.video and values.get('video_url') and v is None:
            raise ValueError('Для видео урока необходимо указать длительность')
        return v

    @validator('theory_content')
    def validate_theory_content(cls, v, values):
        if values.get('lesson_type') == LessonType.theory and not v:
            raise ValueError('Для теоретического урока необходимо указать содержание')
        return v

    @validator('quiz_questions')
    def validate_quiz_questions(cls, v, values):
        if values.get('lesson_type') == LessonType.quiz and not v:
            raise ValueError('Для урока-теста необходимо указать вопросы')
        return v


class LessonCreate(LessonBase):
    """Схема для создания урока"""
    course_id: int = Field(..., description="ID курса")


class LessonUpdate(BaseModel):
    """Схема для обновления урока"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    video_url: Optional[str] = None
    video_duration: Optional[int] = Field(None, ge=0)
    materials: Optional[Dict[str, Any]] = None
    theory_content: Optional[str] = None
    practice_tasks: Optional[Dict[str, Any]] = None
    quiz_questions: Optional[Dict[str, Any]] = None
    order_num: Optional[int] = Field(None, ge=1)
    lesson_type: Optional[LessonType] = None
    is_free: Optional[bool] = None
    is_published: Optional[bool] = None
    min_score: Optional[int] = Field(None, ge=0, le=100)
    max_attempts: Optional[int] = Field(None, ge=1)


class LessonInDBBase(LessonBase):
    """Базовая схема урока в БД"""
    id: int
    course_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Lesson(LessonInDBBase):
    """Схема урока для API ответов"""
    pass


class LessonInDB(LessonInDBBase):
    """Схема урока в базе данных"""
    pass


class LessonWithCourse(Lesson):
    """Схема урока с информацией о курсе"""
    course_title: str
    course_level: str


class LessonListResponse(BaseModel):
    """Схема ответа для списка уроков"""
    lessons: List[Lesson]
    total: int
    page: int
    page_size: int
    total_pages: int


class LessonFilter(BaseModel):
    """Схема для фильтрации уроков"""
    course_id: Optional[int] = None
    lesson_type: Optional[LessonType] = None
    is_free: Optional[bool] = None
    is_published: Optional[bool] = None
    search: Optional[str] = Field(None, description="Поиск по названию и описанию")


class QuizQuestion(BaseModel):
    """Схема вопроса теста"""
    id: str = Field(..., description="ID вопроса")
    question: str = Field(..., description="Текст вопроса")
    type: str = Field(..., description="Тип вопроса (single, multiple, text)")
    options: Optional[List[str]] = Field(None, description="Варианты ответов")
    correct_answers: List[str] = Field(..., description="Правильные ответы")
    points: int = Field(1, ge=1, description="Количество баллов за вопрос")
    explanation: Optional[str] = Field(None, description="Объяснение ответа")


class QuizSubmission(BaseModel):
    """Схема отправки ответов на тест"""
    lesson_id: int = Field(..., description="ID урока")
    answers: Dict[str, List[str]] = Field(..., description="Ответы пользователя")


class QuizResult(BaseModel):
    """Результат прохождения теста"""
    lesson_id: int
    score: int
    max_score: int
    percentage: float
    passed: bool
    attempt_number: int
    time_spent: int
    correct_answers: int
    total_questions: int
    detailed_results: List[Dict[str, Any]]


class LessonProgress(BaseModel):
    """Прогресс по уроку"""
    lesson_id: int
    user_id: int
    status: str
    score: Optional[int] = None
    max_score: Optional[int] = None
    percentage: float = 0.0
    attempts: int = 0
    time_spent: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_accessed_at: datetime