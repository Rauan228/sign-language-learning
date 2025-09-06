from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.models.user import User
from app.schemas.progress import (
    CourseProgress, LessonProgress, CourseProgressCreate, LessonProgressCreate,
    CourseProgressUpdate, LessonProgressUpdate, UserProgressSummary,
    CourseProgressWithDetails, ProgressStatus
)
from app.crud.progress import get_progress_crud
from app.crud.user import get_user_crud
from app.crud.course import get_course_crud
from app.crud.lesson import get_lesson_crud

router = APIRouter()


@router.get("/summary", response_model=UserProgressSummary)
def get_user_progress_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Получить сводку прогресса текущего пользователя"""
    crud = get_progress_crud(db)
    return crud.get_user_progress_summary(current_user.id)


@router.get("/course/{course_id}", response_model=CourseProgressWithDetails)
def get_course_progress(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Получить детальный прогресс по курсу"""
    # Проверяем существование курса
    course_crud = get_course_crud(db)
    course = course_crud.get(course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found"
        )
    
    crud = get_progress_crud(db)
    progress_detail = crud.get_course_progress_detail(
        current_user.id, course_id
    )
    
    if not progress_detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Progress not found"
        )
    
    return progress_detail


@router.get("/lesson/{lesson_id}", response_model=LessonProgress)
def get_lesson_progress(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Получить прогресс по уроку"""
    # Проверяем существование урока
    lesson_crud = get_lesson_crud(db)
    lesson = lesson_crud.get(lesson_id)
    if not lesson:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lesson not found"
        )
    
    crud = get_progress_crud(db)
    progress = crud.get_lesson_progress(current_user.id, lesson_id)
    if not progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Progress not found"
        )
    
    return progress


@router.post("/start-lesson/{lesson_id}", response_model=LessonProgress)
def start_lesson(
    lesson_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Начать урок"""
    # Проверяем существование урока
    lesson_crud = get_lesson_crud(db)
    lesson = lesson_crud.get(lesson_id)
    if not lesson:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lesson not found"
        )
    
    try:
        crud = get_progress_crud(db)
        progress = crud.start_lesson(current_user.id, lesson_id)
        return progress
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/complete-lesson/{lesson_id}", response_model=LessonProgress)
def complete_lesson(
    lesson_id: int,
    score: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Завершить урок"""
    # Проверяем существование урока
    lesson_crud = get_lesson_crud(db)
    lesson = lesson_crud.get(lesson_id)
    if not lesson:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lesson not found"
        )
    
    try:
        crud = get_progress_crud(db)
        progress = crud.complete_lesson(
            current_user.id, lesson_id, 
            score=score
        )
        return progress
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.put("/lesson/{lesson_id}/time", response_model=LessonProgress)
def update_lesson_time(
    lesson_id: int,
    time_delta: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Обновить время изучения урока"""
    # Проверяем существование урока
    lesson_crud = get_lesson_crud(db)
    lesson = lesson_crud.get(lesson_id)
    if not lesson:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lesson not found"
        )
    
    if time_delta < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Time delta must be positive"
        )
    
    crud = get_progress_crud(db)
    progress = crud.update_lesson_time(
        current_user.id, lesson_id, time_delta
    )
    return progress


@router.get("/user/{user_id}/summary", response_model=UserProgressSummary)
def get_user_progress_summary_admin(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Получить сводку прогресса пользователя (для администраторов)"""
    # Проверяем права доступа
    if not current_user.is_superuser and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Проверяем существование пользователя
    user_crud = get_user_crud(db)
    user = user_crud.get_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    crud = get_progress_crud(db)
    return crud.get_user_progress_summary(user_id)


@router.get("/user/{user_id}/course/{course_id}", response_model=CourseProgressWithDetails)
def get_user_course_progress_admin(
    user_id: int,
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Получить прогресс пользователя по курсу (для администраторов)"""
    # Проверяем права доступа
    if not current_user.is_superuser and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Проверяем существование пользователя и курса
    user = crud_user.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    course_crud = get_course_crud(db)
    course = course_crud.get(course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found"
        )
    
    crud = get_progress_crud(db)
    progress_detail = crud.get_course_progress_detail(
        user_id, course_id
    )
    
    if not progress_detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Progress not found"
        )
    
    return progress_detail


@router.get("/", response_model=List[LessonProgress])
def get_user_progress(
    course_id: Optional[int] = None,
    lesson_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Получить весь прогресс пользователя с фильтрацией"""
    crud = get_progress_crud(db)
    progress_list = crud.get_user_progress(
        current_user.id, course_id, lesson_id
    )
    return progress_list


@router.post("/", response_model=LessonProgress)
def create_progress(
    progress_in: LessonProgressCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Создать новый прогресс (для администраторов)"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Проверяем существование связанных объектов
    user_crud = get_user_crud(db)
    user = user_crud.get_by_id(progress_in.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    course_crud = get_course_crud(db)
    course = course_crud.get(progress_in.course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found"
        )
    
    if progress_in.lesson_id:
        lesson_crud = get_lesson_crud(db)
        lesson = lesson_crud.get(progress_in.lesson_id)
        if not lesson:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Lesson not found"
            )
    
    crud = get_progress_crud(db)
    progress = crud.create(obj_in=progress_in)
    return progress


@router.put("/{progress_id}", response_model=LessonProgress)
def update_progress(
    progress_id: int,
    progress_in: LessonProgressUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Обновить прогресс"""
    crud = get_progress_crud(db)
    progress = crud.get(id=progress_id)
    if not progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Progress not found"
        )
    
    # Проверяем права доступа
    if not current_user.is_superuser and progress.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    crud = get_progress_crud(db)
    progress = crud.update(db_obj=progress, obj_in=progress_in)
    return progress


@router.delete("/{progress_id}")
def delete_progress(
    progress_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Удалить прогресс (для администраторов)"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    progress = crud_progress.get(db, id=progress_id)
    if not progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Progress not found"
        )
    
    crud = get_progress_crud(db)
    crud.remove(id=progress_id)
    return {"message": "Progress deleted successfully"}