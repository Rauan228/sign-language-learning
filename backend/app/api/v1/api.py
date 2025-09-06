from fastapi import APIRouter
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.users import router as users_router
from app.api.v1.endpoints.courses import router as courses_router
from app.api.v1.endpoints.lessons import router as lessons_router
from app.api.v1.endpoints.tests import router as tests_router
from app.api.v1.endpoints.certificates import router as certificates_router
from app.api.progress import router as progress_router
from app.api.v1.ai_chat import router as ai_chat_router
from app.api.v1.gesture import router as gesture_router

api_router = APIRouter()

# Подключение роутеров
api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
api_router.include_router(users_router, prefix="/users", tags=["users"])
api_router.include_router(courses_router, prefix="/courses", tags=["courses"])
api_router.include_router(lessons_router, prefix="/lessons", tags=["lessons"])
api_router.include_router(tests_router, prefix="/tests", tags=["tests"])
api_router.include_router(certificates_router, prefix="/certificates", tags=["certificates"])
api_router.include_router(progress_router, prefix="/progress", tags=["progress"])
api_router.include_router(ai_chat_router, prefix="/ai-chat", tags=["ai-chat"])
api_router.include_router(gesture_router, prefix="/gesture", tags=["gesture-recognition"])

# TODO: Добавить остальные роутеры по мере их создания
# api_router.include_router(reviews.router, prefix="/reviews", tags=["reviews"])
# api_router.include_router(admin.router, prefix="/admin", tags=["admin"])