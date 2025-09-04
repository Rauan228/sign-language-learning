from pydantic import BaseModel, EmailStr
from typing import Optional


class Token(BaseModel):
    """Схема токена доступа"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # в секундах


class TokenData(BaseModel):
    """Данные токена"""
    user_id: Optional[int] = None
    email: Optional[str] = None
    role: Optional[str] = None


class RefreshToken(BaseModel):
    """Схема для обновления токена"""
    refresh_token: str


class LoginRequest(BaseModel):
    """Запрос на вход"""
    email: EmailStr
    password: str
    remember_me: bool = False


class LoginResponse(BaseModel):
    """Ответ при входе"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict  # Базовая информация о пользователе


class PasswordResetRequest(BaseModel):
    """Запрос на сброс пароля"""
    email: EmailStr


class PasswordReset(BaseModel):
    """Сброс пароля"""
    token: str
    new_password: str
    confirm_password: str


class ChangePassword(BaseModel):
    """Смена пароля"""
    current_password: str
    new_password: str
    confirm_password: str