from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.core.permissions import require_teacher_or_admin
from app.models.user import User
from app.services.test_service import get_test_service, TestService
from app.schemas.test import (
    Test, TestCreate, TestUpdate, TestAttempt, TestAttemptCreate,
    TestAttemptSubmit, TestResult, TestStatistics, UserTestProgress
)

router = APIRouter()


@router.post("/", response_model=Test, status_code=status.HTTP_201_CREATED)
def create_test(
    test_data: TestCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher_or_admin)
):
    """Создать новый тест"""
    test_service = get_test_service(db)
    return test_service.create_test(test_data, current_user)


@router.get("/{test_id}", response_model=Test)
def get_test(
    test_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Получить тест по ID"""
    test_service = get_test_service(db)
    return test_service.get_test(test_id, current_user)


@router.get("/lesson/{lesson_id}", response_model=List[Test])
def get_tests_by_lesson(
    lesson_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Получить все тесты урока"""
    test_service = get_test_service(db)
    return test_service.get_tests_by_lesson(lesson_id, current_user)


@router.put("/{test_id}", response_model=Test)
def update_test(
    test_id: int,
    test_data: TestUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher_or_admin)
):
    """Обновить тест"""
    test_service = get_test_service(db)
    return test_service.update_test(test_id, test_data, current_user)


@router.delete("/{test_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_test(
    test_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher_or_admin)
):
    """Удалить тест"""
    test_service = get_test_service(db)
    success = test_service.delete_test(test_id, current_user)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Тест не найден"
        )


# Эндпоинты для прохождения тестов
@router.post("/{test_id}/attempts", response_model=TestAttempt, status_code=status.HTTP_201_CREATED)
def start_test_attempt(
    test_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Начать прохождение теста"""
    test_service = get_test_service(db)
    return test_service.start_test_attempt(test_id, current_user)


@router.post("/attempts/{attempt_id}/submit", response_model=TestResult)
def submit_test_attempt(
    attempt_id: int,
    submission: TestAttemptSubmit,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Завершить прохождение теста"""
    test_service = get_test_service(db)
    return test_service.submit_test_attempt(attempt_id, submission, current_user)


@router.get("/{test_id}/attempts", response_model=List[TestAttempt])
def get_user_attempts(
    test_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Получить все попытки пользователя для теста"""
    test_service = get_test_service(db)
    return test_service.get_user_attempts(test_id, current_user)


@router.get("/{test_id}/progress", response_model=UserTestProgress)
def get_user_test_progress(
    test_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Получить прогресс пользователя по тесту"""
    test_service = get_test_service(db)
    return test_service.get_user_test_progress(test_id, current_user)


@router.get("/{test_id}/statistics", response_model=TestStatistics)
def get_test_statistics(
    test_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher_or_admin)
):
    """Получить статистику по тесту (только для преподавателей и админов)"""
    test_service = get_test_service(db)
    return test_service.get_test_statistics(test_id, current_user)


# Дополнительные эндпоинты для работы с вопросами
@router.get("/attempts/{attempt_id}", response_model=TestAttempt)
def get_attempt(
    attempt_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Получить информацию о попытке"""
    test_service = get_test_service(db)
    attempt = test_service.test_crud.get_attempt(attempt_id)
    
    if not attempt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Попытка не найдена"
        )
    
    # Проверяем права доступа
    if attempt.user_id != current_user.id and current_user.role not in ["teacher", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав для просмотра попытки"
        )
    
    return attempt