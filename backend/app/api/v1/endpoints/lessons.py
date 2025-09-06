from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.crud.lesson import get_lesson_crud
from app.services.course import get_course_service
from app.models.user import User
from app.schemas.lesson import (
    Lesson,
    LessonCreate,
    LessonUpdate,
    LessonWithCourse,
    LessonListResponse,
    LessonFilter,
    QuizSubmission,
    QuizResult,
    LessonProgress
)
from app.services.lesson_service import get_lesson_service
from app.core.permissions import require_role, require_teacher_or_admin

router = APIRouter()


@router.post("/", response_model=Lesson, status_code=status.HTTP_201_CREATED)
def create_lesson(
    *,
    db: Session = Depends(get_db),
    lesson_in: LessonCreate,
    current_user: User = Depends(require_teacher_or_admin)
) -> Any:
    """
    Создать новый урок.
    Требует права teacher или admin.
    """
    # Права проверяются через зависимость require_teacher_or_admin
    
    # Проверяем существование курса
    lesson_crud_instance = get_lesson_crud(db)
    course_service = get_course_service(db)
    course = course_service.get_course_by_id(lesson_in.course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Курс не найден"
        )
    
    # Проверяем права на курс (если пользователь teacher)
    if current_user.role == "teacher" and course.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав для создания урока в этом курсе"
        )
    
    # Устанавливаем порядковый номер, если не указан
    if not lesson_in.order_num:
        lesson_in.order_num = lesson_crud_instance.get_next_order_num(
            db=db, course_id=lesson_in.course_id
        )
    
    lesson = lesson_crud_instance.create(db=db, obj_in=lesson_in)
    return lesson


@router.get("/", response_model=LessonListResponse)
def read_lessons(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0, description="Количество пропускаемых записей"),
    limit: int = Query(100, ge=1, le=1000, description="Максимальное количество записей"),
    course_id: Optional[int] = Query(None, description="ID курса"),
    lesson_type: Optional[str] = Query(None, description="Тип урока"),
    is_free: Optional[bool] = Query(None, description="Бесплатный урок"),
    is_published: Optional[bool] = Query(None, description="Опубликованный урок"),
    search: Optional[str] = Query(None, description="Поиск по названию и описанию"),
    current_user: Optional[User] = Depends(get_current_user)
) -> Any:
    """
    Получить список уроков с фильтрацией.
    Публичные уроки доступны всем, приватные - только авторизованным пользователям.
    """
    # Создаем фильтр
    filter_params = LessonFilter(
        course_id=course_id,
        lesson_type=lesson_type,
        is_free=is_free,
        is_published=is_published,
        search=search
    )
    
    # Если пользователь не авторизован, показываем только опубликованные уроки
    if not current_user:
        filter_params.is_published = True
    
    lesson_crud_instance = get_lesson_crud(db)
    lessons = lesson_crud_instance.get_multi_with_filter(
        db=db, filter_params=filter_params, skip=skip, limit=limit
    )
    
    total = lesson_crud_instance.count_with_filter(
        db=db, filter_params=filter_params
    )
    
    return LessonListResponse(
        lessons=lessons,
        total=total,
        skip=skip,
        limit=limit
    )


@router.get("/course/{course_id}", response_model=List[Lesson])
def read_lessons_by_course(
    *,
    db: Session = Depends(get_db),
    course_id: int,
    current_user: Optional[User] = Depends(get_current_user)
) -> Any:
    """
    Получить все уроки курса.
    Неавторизованные пользователи видят только опубликованные уроки.
    """
    # Проверяем существование курса
    course_service = get_course_service(db)
    course = course_service.get_course_by_id(course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Курс не найден"
        )
    
    lesson_crud_instance = get_lesson_crud(db)
    # Если пользователь не авторизован или не имеет прав, показываем только опубликованные
    if not current_user or (
        current_user.role == "student" and 
        course.teacher_id != current_user.id
    ):
        lessons = lesson_crud_instance.get_published_by_course(db=db, course_id=course_id)
    else:
        lessons = lesson_crud_instance.get_by_course(db=db, course_id=course_id)
    
    return lessons


@router.get("/{lesson_id}", response_model=LessonWithCourse)
def read_lesson(
    *,
    db: Session = Depends(get_db),
    lesson_id: int,
    current_user: Optional[User] = Depends(get_current_user)
) -> Any:
    """
    Получить урок по ID.
    """
    lesson_crud_instance = get_lesson_crud(db)
    lesson = lesson_crud_instance.get(db=db, lesson_id=lesson_id)
    if not lesson:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Урок не найден"
        )
    
    # Проверяем доступ к неопубликованному уроку
    if not lesson.is_published:
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Требуется авторизация"
            )
        
        # Только автор курса, админ или преподаватель могут видеть неопубликованные уроки
        if (
            current_user.role == "student" and 
            lesson.course.teacher_id != current_user.id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Недостаточно прав для просмотра этого урока"
            )
    
    return lesson


@router.put("/{lesson_id}", response_model=Lesson)
def update_lesson(
    *,
    db: Session = Depends(get_db),
    lesson_id: int,
    lesson_in: LessonUpdate,
    current_user: User = Depends(require_teacher_or_admin)
) -> Any:
    """
    Обновить урок.
    Требует права teacher или admin.
    """
    lesson_crud_instance = get_lesson_crud(db)
    lesson = lesson_crud_instance.get(db=db, lesson_id=lesson_id)
    if not lesson:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Урок не найден"
        )
    
    # Права проверяются через зависимость require_teacher_or_admin
    
    # Проверяем права на курс (если пользователь teacher)
    if current_user.role == "teacher" and lesson.course.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав для редактирования этого урока"
        )
    
    lesson = lesson_crud_instance.update(db=db, db_obj=lesson, obj_in=lesson_in)
    return lesson


@router.delete("/{lesson_id}", response_model=Lesson)
def delete_lesson(
    *,
    db: Session = Depends(get_db),
    lesson_id: int,
    current_user: User = Depends(require_teacher_or_admin)
) -> Any:
    """
    Удалить урок.
    Требует права teacher или admin.
    """
    lesson_crud_instance = get_lesson_crud(db)
    lesson = lesson_crud_instance.get(db=db, lesson_id=lesson_id)
    if not lesson:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Урок не найден"
        )
    
    # Права проверяются через зависимость require_teacher_or_admin
    
    # Проверяем права на курс (если пользователь teacher)
    if current_user.role == "teacher" and lesson.course.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав для удаления этого урока"
        )
    
    lesson = lesson_crud_instance.delete(db=db, lesson_id=lesson_id)
    return lesson


@router.post("/{lesson_id}/duplicate", response_model=Lesson)
def duplicate_lesson(
    *,
    db: Session = Depends(get_db),
    lesson_id: int,
    new_course_id: Optional[int] = None,
    current_user: User = Depends(require_teacher_or_admin)
) -> Any:
    """
    Дублировать урок.
    Требует права teacher или admin.
    """
    lesson_crud_instance = get_lesson_crud(db)
    lesson = lesson_crud_instance.get(db=db, lesson_id=lesson_id)
    if not lesson:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Урок не найден"
        )
    
    # Права проверяются через зависимость require_teacher_or_admin
    
    # Проверяем права на исходный курс
    if current_user.role == "teacher" and lesson.course.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав для дублирования этого урока"
        )
    
    # Если указан новый курс, проверяем права на него
    if new_course_id:
        course_service = get_course_service(db)
        target_course = course_service.get_course_by_id(new_course_id)
        if not target_course:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Целевой курс не найден"
            )
        
        if current_user.role == "teacher" and target_course.teacher_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Недостаточно прав для добавления урока в целевой курс"
            )
    
    duplicated_lesson = lesson_crud_instance.duplicate_lesson(
        db=db, lesson_id=lesson_id, new_course_id=new_course_id
    )
    
    if not duplicated_lesson:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при дублировании урока"
        )
    
    return duplicated_lesson


@router.put("/course/{course_id}/reorder", response_model=Dict[str, str])
def reorder_lessons(
    *,
    db: Session = Depends(get_db),
    course_id: int,
    lesson_orders: Dict[int, int],
    current_user: User = Depends(require_teacher_or_admin)
) -> Any:
    """
    Изменить порядок уроков в курсе.
    Требует права teacher или admin.
    """
    # Проверяем существование курса
    course_service = get_course_service(db)
    course = course_service.get_course_by_id(course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Курс не найден"
        )
    
    # Права проверяются через зависимость require_teacher_or_admin
    
    # Проверяем права на курс (если пользователь teacher)
    if current_user.role == "teacher" and course.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав для изменения порядка уроков в этом курсе"
        )
    
    lesson_crud_instance = get_lesson_crud(db)
    success = lesson_crud_instance.reorder_lessons(
        db=db, course_id=course_id, lesson_orders=lesson_orders
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при изменении порядка уроков"
        )
    
    return {"message": "Порядок уроков успешно изменен"}


@router.post("/{lesson_id}/quiz/submit", response_model=QuizResult)
def submit_quiz(
    *,
    db: Session = Depends(get_db),
    lesson_id: int,
    submission: QuizSubmission,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Отправить ответы на тест.
    """
    lesson_service = get_lesson_service(db)
    return lesson_service.submit_quiz(
        db=db, lesson_id=lesson_id, submission=submission, user=current_user
    )


@router.get("/{lesson_id}/progress", response_model=LessonProgress)
def get_lesson_progress(
    *,
    db: Session = Depends(get_db),
    lesson_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Получить прогресс пользователя по уроку.
    """
    lesson_service = get_lesson_service(db)
    progress = lesson_service.get_lesson_progress(
        db=db, lesson_id=lesson_id, user=current_user
    )
    
    if not progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Прогресс по уроку не найден"
        )
    
    return progress


@router.post("/{lesson_id}/complete", response_model=LessonProgress)
def mark_lesson_completed(
    *,
    db: Session = Depends(get_db),
    lesson_id: int,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Отметить урок как завершенный.
    """
    lesson_service = get_lesson_service(db)
    return lesson_service.mark_lesson_completed(
        db=db, lesson_id=lesson_id, user=current_user
    )


@router.get("/course/{course_id}/with-progress", response_model=List[Dict[str, Any]])
def get_course_lessons_with_progress(
    *,
    db: Session = Depends(get_db),
    course_id: int,
    current_user: Optional[User] = Depends(get_current_user)
) -> Any:
    """
    Получить уроки курса с информацией о прогрессе.
    """
    lesson_service = get_lesson_service(db)
    return lesson_service.get_course_lessons_with_progress(
        db=db, course_id=course_id, user=current_user
    )