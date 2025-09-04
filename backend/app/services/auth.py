from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from app.core.config import settings
from app.crud.user import get_user_crud
from app.models.user import User
from app.schemas.auth import TokenData

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.user_crud = get_user_crud(db)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Проверка пароля"""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Хеширование пароля"""
        return pwd_context.hash(password)

    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Аутентификация пользователя"""
        user = self.user_crud.get_by_email(email)
        if not user:
            return None
        if not self.verify_password(password, user.password_hash):
            return None
        if not user.is_active:
            return None
        return user

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Создание access токена"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt

    def create_refresh_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Создание refresh токена"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str, token_type: str = "access") -> Optional[TokenData]:
        """Проверка токена"""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            user_id: int = payload.get("sub")
            token_type_payload: str = payload.get("type")
            
            if user_id is None or token_type_payload != token_type:
                return None
            
            token_data = TokenData(user_id=user_id)
            return token_data
        except JWTError:
            return None

    def get_current_user(self, token: str) -> Optional[User]:
        """Получение текущего пользователя по токену"""
        token_data = self.verify_token(token)
        if token_data is None:
            return None
        
        user = self.user_crud.get_by_id(token_data.user_id)
        if user is None or not user.is_active:
            return None
        
        return user

    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Обновление access токена с помощью refresh токена"""
        token_data = self.verify_token(refresh_token, "refresh")
        if token_data is None:
            return None
        
        user = self.user_crud.get_by_id(token_data.user_id)
        if user is None or not user.is_active:
            return None
        
        access_token = self.create_access_token(data={"sub": str(user.id)})
        return access_token

    def create_password_reset_token(self, email: str) -> Optional[str]:
        """Создание токена для сброса пароля"""
        user = self.user_crud.get_by_email(email)
        if not user:
            return None
        
        reset_token_data = {
            "sub": str(user.id),
            "email": user.email,
            "type": "password_reset"
        }
        
        expire = timedelta(hours=settings.PASSWORD_RESET_TOKEN_EXPIRE_HOURS)
        reset_token = self.create_access_token(data=reset_token_data, expires_delta=expire)
        return reset_token

    def verify_password_reset_token(self, token: str) -> Optional[str]:
        """Проверка токена сброса пароля"""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            email: str = payload.get("email")
            token_type: str = payload.get("type")
            
            if email is None or token_type != "password_reset":
                return None
            
            return email
        except JWTError:
            return None

    def reset_password(self, token: str, new_password: str) -> bool:
        """Сброс пароля пользователя"""
        email = self.verify_password_reset_token(token)
        if not email:
            return False
        
        user = self.user_crud.get_by_email(email)
        if not user:
            return False
        
        hashed_password = self.get_password_hash(new_password)
        user.password_hash = hashed_password
        self.db.commit()
        return True

def get_auth_service(db: Session) -> AuthService:
    """Получить экземпляр AuthService"""
    return AuthService(db)