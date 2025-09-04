from fastapi import APIRouter
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.users import router as users_router
from app.api.v1.endpoints.courses import router as courses_router

api_router = APIRouter()

# Подключение роутеров
api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
api_router.include_router(users_router, prefix="/users", tags=["users"])
api_router.include_router(courses_router, prefix="/courses", tags=["courses"])

# TODO: Добавить остальные роутеры по мере их создания
# api_router.include_router(lessons.router, prefix="/lessons", tags=["lessons"])
# api_router.include_router(progress.router, prefix="/progress", tags=["progress"])
# api_router.include_router(ai_chat.router, prefix="/ai-chat", tags=["ai-chat"])
# api_router.include_router(reviews.router, prefix="/reviews", tags=["reviews"])
# api_router.include_router(admin.router, prefix="/admin", tags=["admin"])