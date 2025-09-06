from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.models.user import User
from app.schemas.ai_chat import (
    ChatSessionCreate, ChatSession, Message, MessageListResponse,
    ChatRequest, ChatResponse, FeedbackCreate, ChatStats,
    ChatListResponse, ChatFilter
)
from app.crud.ai_chat import get_ai_chat_crud
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ai-chat", tags=["AI Chat"])


@router.post("/sessions", response_model=ChatSession)
async def create_chat_session(
    session_data: ChatSessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Создать новую сессию AI-чата
    
    - **course_id**: ID курса (опционально)
    - **lesson_id**: ID урока (опционально)
    - **language**: Язык общения (ru/en)
    """
    try:
        crud = get_ai_chat_crud(db)
        session_id = crud.create_session(
            user_id=current_user.id,
            session_data=session_data
        )
        
        logger.info(f"Created chat session {session_id} for user {current_user.id}")
        
        return ChatSessionResponse(
            session_id=session_id,
            created_at=datetime.utcnow(),
            status="active",
            context={
                "course_id": session_data.course_id,
                "lesson_id": session_data.lesson_id,
                "language": session_data.language
            }
        )
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при создании сессии чата"
        )


@router.post("/chat", response_model=ChatResponse)
async def send_message(
    chat_request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Отправить сообщение в AI-чат и получить ответ
    
    - **message**: Текст сообщения
    - **session_id**: ID сессии (если не указан, создается новая)
    - **course_id**: ID текущего курса
    - **lesson_id**: ID текущего урока
    - **gesture_data**: Данные о жесте (для анализа)
    - **attachments**: Прикрепленные файлы
    - **language**: Язык ответа
    """
    try:
        crud = get_ai_chat_crud(db)
        
        # Валидация длины сообщения
        if len(chat_request.message.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Сообщение не может быть пустым"
            )
        
        if len(chat_request.message) > 2000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Сообщение слишком длинное (максимум 2000 символов)"
            )
        
        # Отправляем сообщение и получаем ответ
        chat_message = crud.send_message(
            user_id=current_user.id,
            chat_request=chat_request
        )
        
        logger.info(f"Processed chat message {chat_message.id} for user {current_user.id}")
        
        return ChatResponse(
            id=chat_message.id,
            session_id=chat_message.session_id,
            question=chat_message.question,
            answer=chat_message.answer,
            question_type=chat_message.question_type,
            answer_type=chat_message.answer_type,
            confidence_score=chat_message.confidence_score,
            processing_time=chat_message.processing_time,
            created_at=chat_message.created_at,
            context_data=chat_message.context_data,
            attachments={
                "question_file": chat_message.question_file_url,
                "answer_file": chat_message.answer_file_url
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при обработке сообщения"
        )


@router.get("/sessions", response_model=ChatListResponse)
async def get_user_sessions(
    skip: int = Query(0, ge=0, description="Количество пропускаемых записей"),
    limit: int = Query(20, ge=1, le=100, description="Максимальное количество записей"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Получить список сессий чата пользователя
    """
    try:
        crud = get_ai_chat_crud(db)
        sessions = crud.get_user_sessions(
            user_id=current_user.id,
            skip=skip,
            limit=limit
        )
        
        return SessionListResponse(
            sessions=sessions,
            total=len(sessions),
            skip=skip,
            limit=limit
        )
    except Exception as e:
        logger.error(f"Error getting user sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при получении сессий"
        )


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def get_session_messages(
    session_id: str = Path(..., description="ID сессии чата"),
    skip: int = Query(0, ge=0, description="Количество пропускаемых сообщений"),
    limit: int = Query(50, ge=1, le=100, description="Максимальное количество сообщений"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Получить сообщения конкретной сессии чата
    """
    try:
        crud = get_ai_chat_crud(db)
        messages = crud.get_session_messages(
            session_id=session_id,
            user_id=current_user.id,
            skip=skip,
            limit=limit
        )
        
        message_responses = [
            MessageResponse(
                id=msg.id,
                session_id=msg.session_id,
                question=msg.question,
                answer=msg.answer,
                question_type=msg.question_type,
                answer_type=msg.answer_type,
                confidence_score=msg.confidence_score,
                processing_time=msg.processing_time,
                user_rating=msg.user_rating,
                is_helpful=msg.is_helpful,
                user_feedback=msg.user_feedback,
                created_at=msg.created_at,
                attachments={
                    "question_file": msg.question_file_url,
                    "answer_file": msg.answer_file_url
                }
            )
            for msg in messages
        ]
        
        return MessageListResponse(
            messages=message_responses,
            total=len(message_responses),
            skip=skip,
            limit=limit,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Error getting session messages: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при получении сообщений сессии"
        )


@router.post("/messages/{message_id}/feedback")
async def add_message_feedback(
    message_id: int = Path(..., description="ID сообщения"),
    feedback: FeedbackCreate = ...,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Добавить обратную связь к сообщению AI
    
    - **rating**: Оценка от 1 до 5
    - **is_helpful**: Был ли ответ полезным
    - **comment**: Комментарий пользователя
    """
    try:
        crud = get_ai_chat_crud(db)
        
        # Валидация рейтинга
        if feedback.rating and (feedback.rating < 1 or feedback.rating > 5):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Рейтинг должен быть от 1 до 5"
            )
        
        updated_message = crud.add_feedback(
            message_id=message_id,
            user_id=current_user.id,
            feedback=feedback
        )
        
        logger.info(f"Added feedback to message {message_id} by user {current_user.id}")
        
        return {
            "message": "Обратная связь добавлена",
            "message_id": message_id,
            "rating": updated_message.user_rating,
            "is_helpful": updated_message.is_helpful
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при добавлении обратной связи"
        )


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str = Path(..., description="ID сессии для удаления"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Удалить сессию чата и все связанные сообщения
    """
    try:
        crud = get_ai_chat_crud(db)
        
        success = crud.delete_session(
            session_id=session_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Сессия не найдена"
            )
        
        logger.info(f"Deleted chat session {session_id} for user {current_user.id}")
        
        return {
            "message": "Сессия чата удалена",
            "session_id": session_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при удалении сессии"
        )


@router.get("/stats", response_model=ChatStats)
async def get_chat_statistics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Получить статистику использования AI-чата для текущего пользователя
    """
    try:
        crud = get_ai_chat_crud(db)
        stats = crud.get_chat_stats(user_id=current_user.id)
        
        return ChatStatsResponse(
            total_messages=stats["total_messages"],
            total_sessions=stats["total_sessions"],
            average_rating=stats["average_rating"],
            average_processing_time=stats["average_processing_time"],
            messages_per_session=stats["messages_per_session"],
            message_types=stats["message_types"]
        )
    except Exception as e:
        logger.error(f"Error getting chat statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при получении статистики"
        )


@router.get("/health")
async def chat_health_check():
    """
    Проверка работоспособности AI-чата
    """
    return {
        "status": "healthy",
        "service": "ai-chat",
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "text_chat": True,
            "gesture_analysis": True,
            "file_attachments": True,
            "multi_language": True,
            "context_awareness": True
        }
    }