from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ProgressStatus(str, Enum):
    """Статусы прогресса"""
    not_started = "not_started"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"
    paused = "paused"


class CourseProgressBase(BaseModel):
    """Базовая схема прогресса по курсу"""
    status: ProgressStatus = Field(ProgressStatus.not_started, description="Статус прохождения курса")
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Процент выполнения курса")
    lessons_completed: int = Field(0, ge=0, description="Количество завершенных уроков")
    total_lessons: int = Field(0, ge=0, description="Общее количество уроков в курсе")
    total_time_spent: int = Field(0, ge=0, description="Общее время, потраченное на курс (в секундах)")
    current_lesson_id: Optional[int] = Field(None, description="ID текущего урока")
    last_accessed_at: Optional[datetime] = Field(None, description="Последнее время доступа")
    notes: Optional[str] = Field(None, description="Заметки пользователя")


class CourseProgressCreate(CourseProgressBase):
    """Схема для создания прогресса по курсу"""
    user_id: int = Field(..., description="ID пользователя")
    course_id: int = Field(..., description="ID курса")


class CourseProgressUpdate(BaseModel):
    """Схема для обновления прогресса по курсу"""
    status: Optional[ProgressStatus] = None
    progress_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    lessons_completed: Optional[int] = Field(None, ge=0)
    total_time_spent: Optional[int] = Field(None, ge=0)
    current_lesson_id: Optional[int] = None
    last_accessed_at: Optional[datetime] = None
    notes: Optional[str] = None


class CourseProgressInDBBase(CourseProgressBase):
    """Базовая схема прогресса по курсу в БД"""
    id: int
    user_id: int
    course_id: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CourseProgress(CourseProgressInDBBase):
    """Схема прогресса по курсу для API ответов"""
    pass


class CourseProgressInDB(CourseProgressInDBBase):
    """Схема прогресса по курсу в базе данных"""
    pass


class CourseProgressWithDetails(CourseProgress):
    """Схема прогресса по курсу с деталями"""
    course_title: str
    course_level: str
    teacher_name: str
    lessons_progress: List['LessonProgress']


class LessonProgressBase(BaseModel):
    """Базовая схема прогресса по уроку"""
    status: ProgressStatus = Field(ProgressStatus.not_started, description="Статус прохождения урока")
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Процент выполнения урока")
    time_spent: int = Field(0, ge=0, description="Время, потраченное на урок (в секундах)")
    attempts: int = Field(0, ge=0, description="Количество попыток")
    best_score: Optional[int] = Field(None, ge=0, le=100, description="Лучший результат")
    last_score: Optional[int] = Field(None, ge=0, le=100, description="Последний результат")
    video_progress: float = Field(0.0, ge=0.0, le=100.0, description="Прогресс просмотра видео")
    quiz_passed: bool = Field(False, description="Пройден ли тест")
    notes: Optional[str] = Field(None, description="Заметки пользователя")


class LessonProgressCreate(LessonProgressBase):
    """Схема для создания прогресса по уроку"""
    user_id: int = Field(..., description="ID пользователя")
    lesson_id: int = Field(..., description="ID урока")
    course_id: int = Field(..., description="ID курса")


class LessonProgressUpdate(BaseModel):
    """Схема для обновления прогресса по уроку"""
    status: Optional[ProgressStatus] = None
    progress_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    time_spent: Optional[int] = Field(None, ge=0)
    attempts: Optional[int] = Field(None, ge=0)
    best_score: Optional[int] = Field(None, ge=0, le=100)
    last_score: Optional[int] = Field(None, ge=0, le=100)
    video_progress: Optional[float] = Field(None, ge=0.0, le=100.0)
    quiz_passed: Optional[bool] = None
    notes: Optional[str] = None


class LessonProgressInDBBase(LessonProgressBase):
    """Базовая схема прогресса по уроку в БД"""
    id: int
    user_id: int
    lesson_id: int
    course_id: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    last_accessed_at: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LessonProgress(LessonProgressInDBBase):
    """Схема прогресса по уроку для API ответов"""
    pass


class LessonProgressInDB(LessonProgressInDBBase):
    """Схема прогресса по уроку в базе данных"""
    pass


class LessonProgressWithDetails(LessonProgress):
    """Схема прогресса по уроку с деталями"""
    lesson_title: str
    lesson_type: str
    course_title: str


class UserProgressSummary(BaseModel):
    """Сводка прогресса пользователя"""
    user_id: int
    total_courses: int
    completed_courses: int
    in_progress_courses: int
    total_lessons: int
    completed_lessons: int
    total_time_spent: int
    average_score: float
    certificates_earned: int
    current_streak: int
    longest_streak: int
    last_activity: Optional[datetime]


class CourseProgressStats(BaseModel):
    """Статистика прогресса по курсу"""
    course_id: int
    total_students: int
    completed_students: int
    in_progress_students: int
    average_completion_time: float
    average_score: float
    completion_rate: float
    most_difficult_lesson: Optional[str]
    dropout_points: List[Dict[str, Any]]


class ProgressFilter(BaseModel):
    """Схема для фильтрации прогресса"""
    user_id: Optional[int] = None
    course_id: Optional[int] = None
    lesson_id: Optional[int] = None
    status: Optional[ProgressStatus] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_score: Optional[int] = Field(None, ge=0, le=100)
    max_score: Optional[int] = Field(None, ge=0, le=100)


class ProgressListResponse(BaseModel):
    """Схема ответа для списка прогресса"""
    progress: List[CourseProgress]
    total: int
    page: int
    page_size: int
    total_pages: int


class ActivityLog(BaseModel):
    """Лог активности пользователя"""
    id: int
    user_id: int
    action: str
    resource_type: str
    resource_id: int
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    class Config:
        from_attributes = True


class Certificate(BaseModel):
    """Сертификат о прохождении курса"""
    id: int
    user_id: int
    course_id: int
    certificate_number: str
    issued_at: datetime
    score: int
    completion_time: int
    valid_until: Optional[datetime] = None
    certificate_url: Optional[str] = None
    
    class Config:
        from_attributes = True


class CertificateCreate(BaseModel):
    """Схема для создания сертификата"""
    user_id: int
    course_id: int
    score: int = Field(..., ge=0, le=100)
    completion_time: int = Field(..., ge=0)


# Обновляем forward reference
CourseProgressWithDetails.model_rebuild()