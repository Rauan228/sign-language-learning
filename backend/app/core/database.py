from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from .config import settings

# Создание движка базы данных
engine = create_engine(
    settings.database_url,
    poolclass=StaticPool,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False  # Установить True для отладки SQL запросов
)

# Создание сессии
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовый класс для моделей
Base = declarative_base()


def get_db():
    """Dependency для получения сессии базы данных"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Инициализация базы данных"""
    # Создание всех таблиц
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")


def drop_db():
    """Удаление всех таблиц (для разработки)"""
    Base.metadata.drop_all(bind=engine)
    print("Database tables dropped successfully!")