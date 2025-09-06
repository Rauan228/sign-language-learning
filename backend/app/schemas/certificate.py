from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class CertificateBase(BaseModel):
    """Базовая схема сертификата"""
    title: str = Field(..., min_length=1, max_length=200, description="Название сертификата")
    description: Optional[str] = Field(None, description="Описание сертификата")


class CertificateCreate(CertificateBase):
    """Схема для создания сертификата"""
    user_id: int = Field(..., description="ID пользователя")
    course_id: int = Field(..., description="ID курса")
    total_score: Optional[int] = Field(0, description="Общий балл")
    max_possible_score: Optional[int] = Field(0, description="Максимально возможный балл")
    percentage: Optional[int] = Field(0, ge=0, le=100, description="Процент прохождения")
    total_time_spent: Optional[int] = Field(0, description="Общее время изучения в минутах")


class CertificateUpdate(BaseModel):
    """Схема для обновления сертификата"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    is_active: Optional[bool] = None


class CertificateInDB(CertificateBase):
    """Схема сертификата в базе данных"""
    id: int
    user_id: int
    course_id: int
    certificate_number: str
    completion_date: datetime
    total_score: int
    max_possible_score: int
    percentage: int
    total_time_spent: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class Certificate(CertificateInDB):
    """Полная схема сертификата для API"""
    pass


class CertificateWithDetails(Certificate):
    """Сертификат с дополнительной информацией"""
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    course_title: Optional[str] = None
    course_description: Optional[str] = None


class CertificateList(BaseModel):
    """Список сертификатов с пагинацией"""
    certificates: list[Certificate]
    total: int
    page: int
    size: int
    pages: int


class CertificateStats(BaseModel):
    """Статистика по сертификатам"""
    total_certificates: int
    active_certificates: int
    certificates_this_month: int
    average_score: float
    average_completion_time: int  # В минутах