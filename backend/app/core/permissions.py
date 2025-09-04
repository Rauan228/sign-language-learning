from fastapi import HTTPException, status, Depends
from typing import List
from app.schemas.user import UserResponse
from app.api.v1.endpoints.auth import get_current_user


def require_role(allowed_roles: List[str]):
    """
    Декоратор для проверки роли пользователя
    
    Args:
        allowed_roles: Список разрешенных ролей
    
    Returns:
        Функция зависимости для FastAPI
    """
    def role_checker(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
        if current_user.role.value not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Недостаточно прав. Требуется роль: {', '.join(allowed_roles)}"
            )
        return current_user
    
    return role_checker


def require_admin(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """
    Проверка роли администратора
    """
    if current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Требуются права администратора"
        )
    return current_user


def require_teacher_or_admin(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """
    Проверка роли преподавателя или администратора
    """
    if current_user.role.value not in ["teacher", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Требуются права преподавателя или администратора"
        )
    return current_user