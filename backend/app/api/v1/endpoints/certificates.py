from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.api.v1.endpoints.auth import get_current_user
from app.core.database import get_db
from app.crud.certificate import get_certificate_crud
from app.models.user import User
from app.schemas.certificate import (
    Certificate as CertificateSchema,
    CertificateCreate,
    CertificateUpdate,
    CertificateWithDetails,
    CertificateList,
    CertificateStats
)

router = APIRouter()


@router.post("/", response_model=CertificateSchema, status_code=status.HTTP_201_CREATED)
def create_certificate(
    certificate_in: CertificateCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Создать сертификат.
    
    Требует завершения курса пользователем.
    """
    certificate_crud = get_certificate_crud(db)
    
    try:
        certificate = certificate_crud.create(certificate_in)
        return certificate
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при создании сертификата"
        )


@router.get("/", response_model=List[CertificateList])
def get_certificates(
    skip: int = Query(0, ge=0, description="Количество пропускаемых записей"),
    limit: int = Query(100, ge=1, le=1000, description="Максимальное количество записей"),
    user_id: Optional[int] = Query(None, description="ID пользователя для фильтрации"),
    course_id: Optional[int] = Query(None, description="ID курса для фильтрации"),
    active_only: bool = Query(True, description="Только активные сертификаты"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Получить список сертификатов с фильтрацией.
    """
    certificate_crud = get_certificate_crud(db)
    
    if user_id:
        certificates = certificate_crud.get_user_certificates(
            user_id=user_id,
            skip=skip,
            limit=limit,
            active_only=active_only
        )
    elif course_id:
        certificates = certificate_crud.get_course_certificates(
            course_id=course_id,
            skip=skip,
            limit=limit
        )
    else:
        # Для обычных пользователей показываем только их сертификаты
        if not current_user.is_superuser:
            user_id = current_user.id
        
        certificates = certificate_crud.get_user_certificates(
            user_id=user_id or current_user.id,
            skip=skip,
            limit=limit,
            active_only=active_only
        )
    
    return certificates


@router.get("/my", response_model=List[CertificateList])
def get_my_certificates(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Получить сертификаты текущего пользователя.
    """
    certificate_crud = get_certificate_crud(db)
    
    certificates = certificate_crud.get_user_certificates(
        user_id=current_user.id,
        skip=skip,
        limit=limit,
        active_only=active_only
    )
    
    return certificates


@router.get("/stats", response_model=CertificateStats)
def get_certificate_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Получить статистику по сертификатам.
    
    Доступно только администраторам.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав доступа"
        )
    
    certificate_crud = get_certificate_crud(db)
    return certificate_crud.get_stats()


@router.get("/number/{certificate_number}", response_model=CertificateWithDetails)
def get_certificate_by_number(
    certificate_number: str,
    db: Session = Depends(get_db)
):
    """
    Получить сертификат по номеру (публичный доступ для верификации).
    """
    certificate_crud = get_certificate_crud(db)
    
    certificate = certificate_crud.get_by_number(certificate_number)
    if not certificate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Сертификат не найден"
        )
    
    if not certificate.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Сертификат недействителен"
        )
    
    certificate_details = certificate_crud.get_with_details(certificate.id)
    return certificate_details


@router.get("/{certificate_id}", response_model=CertificateWithDetails)
def get_certificate(
    certificate_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Получить сертификат по ID с подробной информацией.
    """
    certificate_crud = get_certificate_crud(db)
    
    certificate = certificate_crud.get(certificate_id)
    if not certificate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Сертификат не найден"
        )
    
    # Проверяем права доступа
    if not current_user.is_superuser and certificate.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав доступа"
        )
    
    certificate_details = certificate_crud.get_with_details(certificate_id)
    return certificate_details


@router.put("/{certificate_id}", response_model=CertificateSchema)
def update_certificate(
    certificate_id: int,
    certificate_in: CertificateUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Обновить сертификат.
    
    Доступно только администраторам.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав доступа"
        )
    
    certificate_crud = get_certificate_crud(db)
    
    certificate = certificate_crud.update(certificate_id, certificate_in)
    if not certificate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Сертификат не найден"
        )
    
    return certificate


@router.delete("/{certificate_id}", response_model=CertificateSchema)
def deactivate_certificate(
    certificate_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Деактивировать сертификат.
    
    Доступно только администраторам.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав доступа"
        )
    
    certificate_crud = get_certificate_crud(db)
    
    certificate = certificate_crud.deactivate(certificate_id)
    if not certificate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Сертификат не найден"
        )
    
    return certificate


@router.post("/check-eligibility/{user_id}/{course_id}", response_model=dict)
def check_certificate_eligibility(
    user_id: int,
    course_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Проверить возможность выдачи сертификата.
    """
    # Проверяем права доступа
    if not current_user.is_superuser and user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав доступа"
        )
    
    certificate_crud = get_certificate_crud(db)
    
    can_issue = certificate_crud.can_issue_certificate(user_id, course_id)
    
    return {
        "can_issue_certificate": can_issue,
        "user_id": user_id,
        "course_id": course_id
    }


@router.post("/issue/{user_id}/{course_id}", response_model=CertificateSchema)
def issue_certificate(
    user_id: int,
    course_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Выдать сертификат пользователю за завершение курса.
    
    Автоматически определяет параметры на основе прогресса.
    """
    # Проверяем права доступа
    if not current_user.is_superuser and user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав доступа"
        )
    
    certificate_crud = get_certificate_crud(db)
    
    # Создаем данные для сертификата
    certificate_data = CertificateCreate(
        user_id=user_id,
        course_id=course_id,
        title=title,
        description=description
    )
    
    try:
        certificate = certificate_crud.create(certificate_data)
        return certificate
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при выдаче сертификата"
        )