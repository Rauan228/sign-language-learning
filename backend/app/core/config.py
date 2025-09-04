from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Настройки приложения"""
    
    # Основные настройки
    PROJECT_NAME: str = "Visual Mind API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # База данных MySQL
    DATABASE_URL: str = "mysql://root:@127.127.126.50/Berufsberatung"
    
    # MySQL Database Configuration
    MYSQL_HOST: str = "127.127.126.50"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "Berufsberatung"
    
    # JWT настройки
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS настройки
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:8081",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
    ]
    
    # Файлы и загрузки
    UPLOAD_DIR: str = "uploads"
    STATIC_DIR: str = "static"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/gif"]
    ALLOWED_VIDEO_TYPES: List[str] = ["video/mp4", "video/avi", "video/mov"]
    
    # Email настройки (для восстановления пароля)
    SMTP_TLS: bool = True
    SMTP_PORT: int = 587
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAILS_FROM_EMAIL: str = "noreply@visualmind.com"
    EMAILS_FROM_NAME: str = "Visual Mind"
    
    # Пагинация
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    # AI Chat настройки
    AI_MODEL_URL: str = "http://localhost:5000/api/chat"  # URL к AI модели
    AI_TIMEOUT: int = 30
    
    # Логирование
    LOG_LEVEL: str = "INFO"
    
    # Безопасность
    PASSWORD_MIN_LENGTH: int = 8
    
    @property
    def database_url(self) -> str:
        """Формирование URL для подключения к базе данных"""
        return f"mysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Создание экземпляра настроек
settings = Settings()