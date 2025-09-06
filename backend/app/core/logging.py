import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Получить настроенный логгер
    
    Args:
        name: Имя логгера
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Создаем обработчик для вывода в консоль
        handler = logging.StreamHandler(sys.stdout)
        
        # Устанавливаем формат сообщений
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Добавляем обработчик к логгеру
        logger.addHandler(handler)
        
        # Устанавливаем уровень логирования
        if level:
            logger.setLevel(getattr(logging, level.upper()))
        else:
            logger.setLevel(logging.INFO)
    
    return logger