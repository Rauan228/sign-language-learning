from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user

from app.schemas.course import (
    Course,
    CourseCreate,
    CourseUpdate,
    CourseWithTeacher,
    CourseListResponse,
    CourseFilter,
    CourseStats,
    CourseLevel,
    SignLanguage,
    CourseStatus
)
from app.schemas.user import UserResponse
from app.services.course import get_course_service
from app.core.permissions import require_role
from app.core.exceptions import (
    CourseNotFoundError,
    CourseAlreadyExistsError,
    EnrollmentError,
    PermissionDeniedError
)

router = APIRouter()


@router.get("/", response_model=CourseListResponse)
async def get_courses(
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(20, ge=1, le=100, description="Размер страницы"),
    level: Optional[CourseLevel] = Query(None, description="Фильтр по уровню"),
    language: Optional[SignLanguage] = Query(None, description="Фильтр по языку жестов"),
    status: Optional[CourseStatus] = Query(None, description="Фильтр по статусу"),
    teacher_id: Optional[int] = Query(None, description="Фильтр по преподавателю"),
    is_free: Optional[bool] = Query(None, description="Фильтр по бесплатности"),
    search: Optional[str] = Query(None, description="Поиск по названию и описанию"),
    sort_by: str = Query("created_at", description="Поле для сортировки"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Порядок сортировки"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    """Получить список курсов с фильтрацией и пагинацией"""
    try:
        course_service = get_course_service(db)
        
        # Создаем фильтр
        filter_params = CourseFilter(
            level=level,
            language=language,
            status=status,
            teacher_id=teacher_id,
            is_free=is_free,
            search=search,
            include_unpublished=current_user.is_admin or current_user.is_teacher
        )
        
        return await course_service.get_courses(
            filter_params=filter_params,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении курсов: {str(e)}"
        )


@router.get("/my", response_model=CourseListResponse)
async def get_my_courses(
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(20, ge=1, le=100, description="Размер страницы"),
    status: Optional[CourseStatus] = Query(None, description="Фильтр по статусу"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    """Получить курсы текущего пользователя (для преподавателей - созданные, для студентов - записанные)"""
    try:
        course_service = get_course_service(db)
        
        if current_user.is_teacher or current_user.is_admin:
            # Для преподавателей - показываем созданные курсы
            filter_params = CourseFilter(
                teacher_id=current_user.id,
                status=status,
                include_unpublished=True
            )
            return await course_service.get_courses(
                filter_params=filter_params,
                page=page,
                page_size=page_size
            )
        else:
            # Для студентов - показываем записанные курсы
            return await course_service.get_enrolled_courses(
                user_id=current_user.id,
                page=page,
                page_size=page_size
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении курсов: {str(e)}"
        )


@router.get("/enrolled", response_model=CourseListResponse)
async def get_enrolled_courses(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    """Получить курсы, на которые записан пользователь"""
    course_service = get_course_service(db)
    return await course_service.get_enrolled_courses(
        user_id=current_user.id,
        page=page,
        page_size=page_size
    )


@router.get("/featured", response_model=List[CourseWithTeacher])
async def get_featured_courses(
    limit: int = Query(6, ge=1, le=20, description="Количество курсов"),
    db: Session = Depends(get_db)
):
    """Получить рекомендуемые курсы"""
    course_service = get_course_service(db)
    return await course_service.get_featured_courses(limit=limit)


@router.get("/popular", response_model=List[CourseWithTeacher])
async def get_popular_courses(
    limit: int = Query(6, ge=1, le=20),
    db: Session = Depends(get_db)
):
    """Получить популярные курсы"""
    course_service = get_course_service(db)
    return await course_service.get_popular_courses(limit=limit)


@router.get("/stats", response_model=CourseStats)
async def get_course_stats(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(require_role(["admin"]))
):
    """Получить общую статистику курсов (только для администраторов)"""
    course_service = get_course_service(db)
    return await course_service.get_course_stats()


@router.get("/{course_id}", response_model=CourseWithTeacher)
async def get_course(
    course_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    """Получить курс по ID"""
    try:
        course_service = get_course_service(db)
        course = await course_service.get_course_by_id(course_id)
        
        if not course:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Курс не найден"
            )
        
        return course
    except CourseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Курс не найден"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении курса: {str(e)}"
        )


@router.post("/", response_model=Course, status_code=status.HTTP_201_CREATED)
async def create_course(
    course_data: CourseCreate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(require_role(["teacher", "admin"]))
):
    """Создать новый курс"""
    try:
        course_service = get_course_service(db)
        
        # Если не админ, то курс может создать только для себя
        if not current_user.is_admin and course_data.teacher_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Вы можете создавать курсы только для себя"
            )
        
        course = await course_service.create_course(course_data)
        return course
        
    except CourseAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при создании курса: {str(e)}"
        )


@router.put("/{course_id}", response_model=Course)
async def update_course(
    course_id: int,
    course_data: CourseUpdate,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(require_role(["teacher", "admin"]))
):
    """Обновить курс"""
    try:
        course_service = get_course_service(db)
        
        # Проверяем права доступа
        existing_course = await course_service.get_course_by_id(course_id)
        if not existing_course:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Курс не найден"
            )
        
        # Только админ или автор курса может его редактировать
        if not current_user.is_admin and existing_course.teacher_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="У вас нет прав для редактирования этого курса"
            )
        
        course = await course_service.update_course(course_id, course_data)
        return course
        
    except CourseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Курс не найден"
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обновлении курса: {str(e)}"
        )


@router.delete("/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course(
    course_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(require_role(["teacher", "admin"]))
):
    """Удалить курс"""
    try:
        course_service = get_course_service(db)
        
        # Проверяем права доступа
        existing_course = await course_service.get_course_by_id(course_id)
        if not existing_course:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Курс не найден"
            )
        
        # Только админ или автор курса может его удалить
        if not current_user.is_admin and existing_course.teacher_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="У вас нет прав для удаления этого курса"
            )
        
        await course_service.delete_course(course_id)
        
    except CourseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Курс не найден"
        )
    except EnrollmentError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при удалении курса: {str(e)}"
        )


@router.post("/{course_id}/enroll", status_code=status.HTTP_201_CREATED)
async def enroll_in_course(
    course_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    """Записаться на курс"""
    try:
        course_service = get_course_service(db)
        
        # Проверяем существование курса
        course = await course_service.get_course_by_id(course_id)
        if not course:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Курс не найден"
            )
        
        if not course.is_published:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Нельзя записаться на неопубликованный курс"
            )
        
        enrollment = await course_service.enroll_user(course_id, current_user.id)
        
        return {
            "message": "Успешно записались на курс",
            "enrollment_id": enrollment.id,
            "enrolled_at": enrollment.enrolled_at
        }
        
    except CourseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Курс не найден"
        )
    except EnrollmentError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при записи на курс: {str(e)}"
        )


@router.delete("/{course_id}/enroll", status_code=status.HTTP_204_NO_CONTENT)
async def unenroll_from_course(
    course_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    """Отписаться от курса"""
    try:
        course_service = get_course_service(db)
        
        # Проверяем существование курса
        course = await course_service.get_course_by_id(course_id)
        if not course:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Курс не найден"
            )
        
        await course_service.unenroll_user(course_id, current_user.id)
        
    except CourseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Курс не найден"
        )
    except EnrollmentError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при отписке от курса: {str(e)}"
        )


@router.get("/{course_id}/students", response_model=List[dict])
async def get_course_students(
    course_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    """Получить список студентов курса"""
    try:
        course_service = get_course_service(db)
        
        # Проверяем права доступа
        course = await course_service.get_course_by_id(course_id)
        if not current_user.is_admin and course.teacher_id != current_user.id:
            raise PermissionDeniedError("Доступ разрешен только преподавателю курса или администратору")
        
        return await course_service.get_course_students(
            course_id=course_id,
            page=page,
            page_size=page_size
        )
        
    except CourseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Курс не найден"
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении студентов: {str(e)}"
        )


@router.patch("/{course_id}/publish", response_model=Course)
async def publish_course(
    course_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    """Опубликовать курс"""
    try:
        course_service = get_course_service(db)
        
        # Проверяем права доступа
        course = await course_service.get_course_by_id(course_id)
        if not current_user.is_admin and course.teacher_id != current_user.id:
            raise PermissionDeniedError("Только автор курса или администратор может публиковать курс")
        
        return await course_service.publish_course(course_id)
        
    except CourseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Курс не найден"
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при публикации курса: {str(e)}"
        )


@router.patch("/{course_id}/unpublish", response_model=Course)
async def unpublish_course(
    course_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    """Снять курс с публикации"""
    try:
        course_service = get_course_service(db)
        
        # Проверяем права доступа
        course = await course_service.get_course_by_id(course_id)
        if not current_user.is_admin and course.teacher_id != current_user.id:
            raise PermissionDeniedError("Только автор курса или администратор может снимать курс с публикации")
        
        return await course_service.unpublish_course(course_id)
        
    except CourseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Курс не найден"
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при снятии курса с публикации: {str(e)}"
        )


@router.get("/{course_id}/stats", response_model=dict)
async def get_course_detailed_stats(
    course_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_user)
):
    """Получить детальную статистику курса"""
    try:
        course_service = get_course_service(db)
        
        # Проверяем права доступа
        course = await course_service.get_course_by_id(course_id)
        if not current_user.is_admin and course.teacher_id != current_user.id:
            raise PermissionDeniedError("Только автор курса или администратор может просматривать статистику")
        
        return await course_service.get_course_detailed_stats(course_id)
        
    except CourseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Курс не найден"
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении статистики курса: {str(e)}"
        )