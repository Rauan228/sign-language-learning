from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.user import UserResponse, UserUpdate, UserList, UserStats
from app.crud.user import get_user_crud
from app.api.v1.endpoints.auth import get_current_user
from app.models.user import User, UserRoleEnum

router = APIRouter()

def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Проверка прав администратора"""
    if current_user.role != UserRoleEnum.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

def require_teacher_or_admin(current_user: User = Depends(get_current_user)) -> User:
    """Проверка прав учителя или администратора"""
    if current_user.role not in [UserRoleEnum.TEACHER, UserRoleEnum.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

@router.get("/", response_model=UserList)
async def get_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    role: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Получить список пользователей (только для администраторов)"""
    user_crud = get_user_crud(db)
    
    users = user_crud.get_multi(
        skip=skip,
        limit=limit,
        role=role,
        is_active=is_active,
        search=search
    )
    
    total = user_crud.count_total()
    
    return UserList(
        users=[UserResponse.from_orm(user) for user in users],
        total=total,
        skip=skip,
        limit=limit
    )

@router.get("/stats", response_model=UserStats)
async def get_user_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Получить статистику пользователей (только для администраторов)"""
    user_crud = get_user_crud(db)
    
    total_users = user_crud.count_total()
    students_count = user_crud.count_by_role(UserRoleEnum.STUDENT.value)
    teachers_count = user_crud.count_by_role(UserRoleEnum.TEACHER.value)
    admins_count = user_crud.count_by_role(UserRoleEnum.ADMIN.value)
    
    return UserStats(
        total_users=total_users,
        students_count=students_count,
        teachers_count=teachers_count,
        admins_count=admins_count
    )

@router.get("/teachers", response_model=List[UserResponse])
async def get_teachers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Получить список учителей (доступно всем авторизованным пользователям)"""
    user_crud = get_user_crud(db)
    
    teachers = user_crud.get_multi(
        skip=skip,
        limit=limit,
        role=UserRoleEnum.TEACHER.value,
        is_active=True
    )
    
    return [UserResponse.from_orm(teacher) for teacher in teachers]

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Получить пользователя по ID"""
    user_crud = get_user_crud(db)
    
    # Пользователь может просматривать только свой профиль или админ может просматривать любой
    if current_user.id != user_id and current_user.role != UserRoleEnum.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    user = user_crud.get_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.from_orm(user)

@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Обновить пользователя"""
    user_crud = get_user_crud(db)
    
    # Пользователь может редактировать только свой профиль или админ может редактировать любой
    if current_user.id != user_id and current_user.role != UserRoleEnum.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Проверяем, что пользователь существует
    existing_user = user_crud.get_by_id(user_id)
    if not existing_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Обычные пользователи не могут изменять роль
    if current_user.role != UserRoleEnum.ADMIN and user_update.role is not None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot change user role"
        )
    
    updated_user = user_crud.update(user_id, user_update)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating user"
        )
    
    return UserResponse.from_orm(updated_user)

@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Удалить пользователя (только для администраторов)"""
    user_crud = get_user_crud(db)
    
    # Админ не может удалить сам себя
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete yourself"
        )
    
    success = user_crud.delete(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": "User deleted successfully"}

@router.post("/{user_id}/activate")
async def activate_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Активировать пользователя (только для администраторов)"""
    user_crud = get_user_crud(db)
    
    user = user_crud.activate(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": "User activated successfully"}

@router.post("/{user_id}/deactivate")
async def deactivate_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Деактивировать пользователя (только для администраторов)"""
    user_crud = get_user_crud(db)
    
    # Админ не может деактивировать сам себя
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate yourself"
        )
    
    user = user_crud.deactivate(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": "User deactivated successfully"}

@router.post("/{user_id}/verify-email")
async def verify_user_email(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Подтвердить email пользователя (только для администраторов)"""
    user_crud = get_user_crud(db)
    
    user = user_crud.verify_email(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": "User email verified successfully"}