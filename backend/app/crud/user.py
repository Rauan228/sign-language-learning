from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from passlib.context import CryptContext
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserCRUD:
    def __init__(self, db: Session):
        self.db = db

    def get_password_hash(self, password: str) -> str:
        """Хеширование пароля"""
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Проверка пароля"""
        return pwd_context.verify(plain_password, hashed_password)

    def get_by_id(self, user_id: int) -> Optional[User]:
        """Получить пользователя по ID"""
        return self.db.query(User).filter(User.id == user_id).first()

    def get_by_email(self, email: str) -> Optional[User]:
        """Получить пользователя по email"""
        return self.db.query(User).filter(User.email == email).first()

    def get_multi(
        self, 
        skip: int = 0, 
        limit: int = 100,
        role: Optional[str] = None,
        is_active: Optional[bool] = None,
        search: Optional[str] = None
    ) -> List[User]:
        """Получить список пользователей с фильтрацией"""
        query = self.db.query(User)
        
        if role:
            query = query.filter(User.role == role)
        
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        
        if search:
            query = query.filter(
                or_(
                    User.name.contains(search),
                    User.email.contains(search)
                )
            )
        
        return query.offset(skip).limit(limit).all()

    def create(self, user_create: UserCreate) -> User:
        """Создать нового пользователя"""
        hashed_password = self.get_password_hash(user_create.password)
        
        db_user = User(
            name=user_create.name,
            email=user_create.email,
            password_hash=hashed_password,
            sign_language=user_create.sign_language,
            role=user_create.role,
            phone=user_create.phone,
            date_of_birth=user_create.date_of_birth,
            location=user_create.location,
            bio=user_create.bio
        )
        
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    def update(self, user_id: int, user_update: UserUpdate) -> Optional[User]:
        """Обновить пользователя"""
        db_user = self.get_by_id(user_id)
        if not db_user:
            return None
        
        update_data = user_update.dict(exclude_unset=True)
        
        # Если обновляется пароль, хешируем его
        if "password" in update_data:
            update_data["password_hash"] = self.get_password_hash(update_data.pop("password"))
        
        for field, value in update_data.items():
            setattr(db_user, field, value)
        
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    def delete(self, user_id: int) -> bool:
        """Удалить пользователя"""
        db_user = self.get_by_id(user_id)
        if not db_user:
            return False
        
        self.db.delete(db_user)
        self.db.commit()
        return True

    def activate(self, user_id: int) -> Optional[User]:
        """Активировать пользователя"""
        db_user = self.get_by_id(user_id)
        if not db_user:
            return None
        
        db_user.is_active = True
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    def deactivate(self, user_id: int) -> Optional[User]:
        """Деактивировать пользователя"""
        db_user = self.get_by_id(user_id)
        if not db_user:
            return None
        
        db_user.is_active = False
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    def verify_email(self, user_id: int) -> Optional[User]:
        """Подтвердить email пользователя"""
        db_user = self.get_by_id(user_id)
        if not db_user:
            return None
        
        db_user.is_verified = True
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    def count_total(self) -> int:
        """Получить общее количество пользователей"""
        return self.db.query(User).count()

    def count_by_role(self, role: str) -> int:
        """Получить количество пользователей по роли"""
        return self.db.query(User).filter(User.role == role).count()

    def authenticate(self, email: str, password: str) -> Optional[User]:
        """Аутентификация пользователя"""
        user = self.get_by_email(email)
        if not user:
            return None
        if not self.verify_password(password, user.password_hash):
            return None
        return user

def get_user_crud(db: Session) -> UserCRUD:
    """Получить экземпляр UserCRUD"""
    return UserCRUD(db)