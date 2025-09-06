#!/usr/bin/env python3
"""
Скрипт для удаления таблиц тестов из базы данных
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text
from app.core.database import engine

def drop_test_tables():
    """Удаляет таблицы тестов в правильном порядке"""
    
    tables_to_drop = [
        'answers',
        'test_attempts', 
        'question_options',
        'questions',
        'tests'
    ]
    
    with engine.connect() as connection:
        # Отключаем проверку foreign key
        connection.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        
        for table in tables_to_drop:
            try:
                connection.execute(text(f"DROP TABLE IF EXISTS {table}"))
                print(f"Таблица {table} удалена")
            except Exception as e:
                print(f"Ошибка при удалении таблицы {table}: {e}")
        
        # Включаем обратно проверку foreign key
        connection.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        connection.commit()
        
    print("Все таблицы тестов удалены")

if __name__ == "__main__":
    drop_test_tables()