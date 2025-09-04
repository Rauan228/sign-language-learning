from pydantic import BaseModel, EmailStr, validator
from typing import Optional
from datetime import datetime
from app.models.user import SignLanguageEnum, UserRoleEnum


class UserBase(BaseModel):
    """Базовая схема пользователя"""
    name: str
    email: EmailStr
    phone: Optional[str] = None
    sign_language: SignLanguageEnum = SignLanguageEnum.RU
    
    @validator('name')
    def validate_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Имя должно содержать минимум 2 символа')
        return v.strip()
    
    @validator('phone')
    def validate_phone(cls, v):
        if v and len(v.strip()) < 10:
            raise ValueError('Номер телефона должен содержать минимум 10 цифр')
        return v.strip() if v else None


class UserCreate(UserBase):
    """Схема для создания пользователя"""
    password: str
    confirm_password: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Пароль должен содержать минимум 8 символов')
        if not any(c.isupper() for c in v):
            raise ValueError('Пароль должен содержать хотя бы одну заглавную букву')
        if not any(c.islower() for c in v):
            raise ValueError('Пароль должен содержать хотя бы одну строчную букву')
        if not any(c.isdigit() for c in v):
            raise ValueError('Пароль должен содержать хотя бы одну цифру')
        return v
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Пароли не совпадают')
        return v


class UserUpdate(BaseModel):
    """Схема для обновления пользователя"""
    name: Optional[str] = None
    phone: Optional[str] = None
    sign_language: Optional[SignLanguageEnum] = None
    avatar_url: Optional[str] = None
    
    @validator('name')
    def validate_name(cls, v):
        if v and len(v.strip()) < 2:
            raise ValueError('Имя должно содержать минимум 2 символа')
        return v.strip() if v else None


class UserResponse(UserBase):
    """Схема ответа с данными пользователя"""
    id: int
    role: UserRoleEnum
    is_active: bool
    is_verified: bool
    avatar_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UserProfile(UserResponse):
    """Расширенная схема профиля пользователя"""
    # Статистика
    courses_count: Optional[int] = 0
    completed_courses: Optional[int] = 0
    total_study_time: Optional[int] = 0  # в минутах
    certificates_count: Optional[int] = 0
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """Схема для входа пользователя"""
    email: EmailStr
    password: str
    remember_me: bool = False


class UserList(BaseModel):
    """Схема для списка пользователей"""
    users: list[UserResponse]
    total: int
    page: int
    size: int
    pages: int


class UserStats(BaseModel):
    """Статистика пользователя"""
    total_users: int
    active_users: int
    new_users_today: int
    new_users_week: int
    students_count: int
    teachers_count: int
    admins_count: int


class PasswordResetRequest(BaseModel):
    """Запрос на восстановление пароля"""
    email: EmailStr


class PasswordReset(BaseModel):
    """Сброс пароля"""
    token: str
    new_password: str
    confirm_password: str
    
    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Пароль должен содержать минимум 8 символов')
        return v
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Пароли не совпадают')
        return v


class UserSettings(BaseModel):
    """Настройки пользователя"""
    sign_language: Optional[SignLanguageEnum] = None
    email_notifications: bool = True
    push_notifications: bool = True
    theme: str = "light"  # light, dark, auto
    language: str = "ru"  # ru, en, kz