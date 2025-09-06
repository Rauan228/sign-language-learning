from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
from datetime import datetime, timedelta
import uuid

from app.models.ai_chat import AIChat, MessageTypeEnum, ResponseTypeEnum
from app.models.user import User
from app.models.course import Course
from app.models.lesson import Lesson
from app.schemas.ai_chat import (
    ChatSessionCreate, ChatSessionUpdate, MessageCreate, MessageUpdate,
    ChatRequest, FeedbackCreate, ChatFilter
)


class CRUDAIChat:
    """CRUD операции для AI чата"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def _generate_session_id(self) -> str:
        """Генерация уникального ID сессии"""
        return str(uuid.uuid4())
    
    def create_session(self, user_id: int, session_data: ChatSessionCreate) -> str:
        """Создать новую сессию чата"""
        session_id = self._generate_session_id()
        
        # Создаем системное сообщение для начала сессии
        system_message = self._create_system_message(
            user_id=user_id,
            session_id=session_id,
            course_id=session_data.course_id,
            lesson_id=session_data.lesson_id,
            language=session_data.language
        )
        
        self.db.add(system_message)
        self.db.commit()
        
        return session_id
    
    def _create_system_message(
        self, 
        user_id: int, 
        session_id: str, 
        course_id: Optional[int] = None,
        lesson_id: Optional[int] = None,
        language: str = "ru"
    ) -> AIChat:
        """Создать системное сообщение для инициализации контекста"""
        context_data = {
            "session_type": "ai_chat",
            "language": language,
            "initialized_at": datetime.utcnow().isoformat()
        }
        
        if course_id:
            course = self.db.query(Course).filter(Course.id == course_id).first()
            if course:
                context_data["course"] = {
                    "id": course.id,
                    "title": course.title,
                    "description": course.description
                }
        
        if lesson_id:
            lesson = self.db.query(Lesson).filter(Lesson.id == lesson_id).first()
            if lesson:
                context_data["lesson"] = {
                    "id": lesson.id,
                    "title": lesson.title,
                    "content": lesson.content[:500] if lesson.content else None
                }
        
        system_prompt = self._get_system_prompt(language, context_data)
        
        return AIChat(
            user_id=user_id,
            session_id=session_id,
            question="[SYSTEM_INIT]",
            question_type=MessageTypeEnum.TEXT,
            answer=system_prompt,
            answer_type=ResponseTypeEnum.TEXT,
            context_data=context_data,
            confidence_score=100,
            processing_time=0
        )
    
    def _get_system_prompt(self, language: str, context: Dict[str, Any]) -> str:
        """Получить системный промпт в зависимости от языка и контекста"""
        if language == "ru":
            prompt = "Привет! Я AI-ассистент для изучения жестового языка. "
            if "course" in context:
                prompt += f"Мы изучаем курс '{context['course']['title']}'. "
            if "lesson" in context:
                prompt += f"Сейчас проходим урок '{context['lesson']['title']}'. "
            prompt += "Я помогу вам с вопросами по жестовому языку, объясню жесты и помогу с практикой. Задавайте любые вопросы!"
        else:
            prompt = "Hello! I'm an AI assistant for sign language learning. "
            if "course" in context:
                prompt += f"We're studying the course '{context['course']['title']}'. "
            if "lesson" in context:
                prompt += f"Currently working on lesson '{context['lesson']['title']}'. "
            prompt += "I'll help you with sign language questions, explain gestures, and assist with practice. Ask me anything!"
        
        return prompt
    
    def send_message(self, user_id: int, chat_request: ChatRequest) -> AIChat:
        """Отправить сообщение и получить ответ от AI"""
        # Если сессия не указана, создаем новую
        if not chat_request.session_id:
            session_data = ChatSessionCreate(
                course_id=chat_request.course_id,
                lesson_id=chat_request.lesson_id,
                language=chat_request.language
            )
            session_id = self.create_session(user_id, session_data)
        else:
            session_id = str(chat_request.session_id)
        
        # Определяем тип сообщения
        message_type = MessageTypeEnum.TEXT
        if chat_request.gesture_data:
            message_type = MessageTypeEnum.GESTURE
        elif chat_request.attachments:
            message_type = MessageTypeEnum.IMAGE
        
        # Получаем контекст сессии
        context_data = self._build_context(
            session_id=session_id,
            course_id=chat_request.course_id,
            lesson_id=chat_request.lesson_id,
            additional_context=chat_request.context
        )
        
        # Генерируем ответ AI (заглушка - в реальности здесь будет вызов OpenAI API)
        ai_response = self._generate_ai_response(
            message=chat_request.message,
            context=context_data,
            language=chat_request.language,
            gesture_data=chat_request.gesture_data
        )
        
        # Создаем запись в базе данных
        chat_message = AIChat(
            user_id=user_id,
            session_id=session_id,
            question=chat_request.message,
            question_type=message_type,
            question_file_url=chat_request.attachments[0] if chat_request.attachments else None,
            answer=ai_response["text"],
            answer_type=ResponseTypeEnum.TEXT,
            answer_file_url=ai_response.get("file_url"),
            context_data=context_data,
            confidence_score=ai_response["confidence"],
            processing_time=ai_response["processing_time"]
        )
        
        self.db.add(chat_message)
        self.db.commit()
        self.db.refresh(chat_message)
        
        return chat_message
    
    def _build_context(
        self, 
        session_id: str, 
        course_id: Optional[int] = None,
        lesson_id: Optional[int] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Построить контекст для AI"""
        context = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Получаем историю сообщений (последние 10)
        recent_messages = self.db.query(AIChat).filter(
            and_(
                AIChat.session_id == session_id,
                AIChat.question != "[SYSTEM_INIT]"
            )
        ).order_by(desc(AIChat.created_at)).limit(10).all()
        
        context["message_history"] = [
            {
                "question": msg.question,
                "answer": msg.answer,
                "timestamp": msg.created_at.isoformat()
            }
            for msg in reversed(recent_messages)
        ]
        
        # Добавляем информацию о курсе и уроке
        if course_id:
            course = self.db.query(Course).filter(Course.id == course_id).first()
            if course:
                context["current_course"] = {
                    "id": course.id,
                    "title": course.title,
                    "level": course.level,
                    "language": course.sign_language
                }
        
        if lesson_id:
            lesson = self.db.query(Lesson).filter(Lesson.id == lesson_id).first()
            if lesson:
                context["current_lesson"] = {
                    "id": lesson.id,
                    "title": lesson.title,
                    "type": lesson.lesson_type,
                    "order": lesson.order_index
                }
        
        # Добавляем дополнительный контекст
        if additional_context:
            context.update(additional_context)
        
        return context
    
    def _generate_ai_response(
        self, 
        message: str, 
        context: Dict[str, Any],
        language: str = "ru",
        gesture_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Генерация ответа AI (заглушка для интеграции с OpenAI)"""
        import time
        import random
        
        start_time = time.time()
        
        # Заглушка для демонстрации
        if gesture_data:
            if language == "ru":
                response_text = f"Я вижу, что вы показали жест. Это интересный жест! Попробуйте выполнить его медленнее для лучшего понимания."
            else:
                response_text = f"I can see you performed a gesture. That's an interesting gesture! Try performing it slower for better understanding."
        else:
            if "привет" in message.lower() or "hello" in message.lower():
                if language == "ru":
                    response_text = "Привет! Рад помочь вам с изучением жестового языка. О чем хотите узнать?"
                else:
                    response_text = "Hello! I'm happy to help you learn sign language. What would you like to know?"
            elif "жест" in message.lower() or "gesture" in message.lower():
                if language == "ru":
                    response_text = "Жесты - это основа жестового языка. Каждый жест имеет свое значение и правильную технику выполнения. Какой конкретный жест вас интересует?"
                else:
                    response_text = "Gestures are the foundation of sign language. Each gesture has its meaning and proper execution technique. Which specific gesture interests you?"
            else:
                if language == "ru":
                    response_text = f"Спасибо за ваш вопрос: '{message}'. Я постараюсь помочь вам с изучением жестового языка. Можете задать более конкретный вопрос?"
                else:
                    response_text = f"Thank you for your question: '{message}'. I'll try to help you with sign language learning. Could you ask a more specific question?"
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "text": response_text,
            "confidence": random.randint(85, 98),
            "processing_time": processing_time,
            "file_url": None
        }
    
    def get_session_messages(
        self, 
        session_id: str, 
        user_id: int,
        skip: int = 0, 
        limit: int = 50
    ) -> List[AIChat]:
        """Получить сообщения сессии"""
        return self.db.query(AIChat).filter(
            and_(
                AIChat.session_id == session_id,
                AIChat.user_id == user_id,
                AIChat.question != "[SYSTEM_INIT]"
            )
        ).order_by(AIChat.created_at).offset(skip).limit(limit).all()
    
    def get_user_sessions(
        self, 
        user_id: int, 
        skip: int = 0, 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Получить сессии пользователя"""
        sessions = self.db.query(
            AIChat.session_id,
            func.min(AIChat.created_at).label('started_at'),
            func.max(AIChat.created_at).label('last_message_at'),
            func.count(AIChat.id).label('message_count')
        ).filter(
            AIChat.user_id == user_id
        ).group_by(
            AIChat.session_id
        ).order_by(
            desc(func.max(AIChat.created_at))
        ).offset(skip).limit(limit).all()
        
        result = []
        for session in sessions:
            # Получаем первое сообщение для контекста
            first_message = self.db.query(AIChat).filter(
                and_(
                    AIChat.session_id == session.session_id,
                    AIChat.user_id == user_id
                )
            ).order_by(AIChat.created_at).first()
            
            result.append({
                "session_id": session.session_id,
                "started_at": session.started_at,
                "last_message_at": session.last_message_at,
                "message_count": session.message_count - 1,  # Исключаем системное сообщение
                "context": first_message.context_data if first_message else {},
                "title": self._generate_session_title(first_message)
            })
        
        return result
    
    def _generate_session_title(self, first_message: Optional[AIChat]) -> str:
        """Генерация заголовка сессии"""
        if not first_message or not first_message.context_data:
            return "Общий чат"
        
        context = first_message.context_data
        if "course" in context:
            course_title = context["course"].get("title", "Неизвестный курс")
            if "lesson" in context:
                lesson_title = context["lesson"].get("title", "Неизвестный урок")
                return f"{course_title} - {lesson_title}"
            return course_title
        
        return "Общий чат"
    
    def add_feedback(self, message_id: int, user_id: int, feedback: FeedbackCreate) -> AIChat:
        """Добавить обратную связь к сообщению"""
        message = self.db.query(AIChat).filter(
            and_(
                AIChat.id == message_id,
                AIChat.user_id == user_id
            )
        ).first()
        
        if not message:
            raise ValueError("Сообщение не найдено")
        
        message.user_rating = feedback.rating
        message.is_helpful = "yes" if feedback.is_helpful else "no"
        message.user_feedback = feedback.comment
        
        self.db.commit()
        self.db.refresh(message)
        
        return message
    
    def get_chat_stats(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Получить статистику чатов"""
        query = self.db.query(AIChat)
        if user_id:
            query = query.filter(AIChat.user_id == user_id)
        
        total_messages = query.count()
        
        # Количество сессий
        sessions_query = query.with_entities(AIChat.session_id).distinct()
        total_sessions = sessions_query.count()
        
        # Средняя оценка
        avg_rating = self.db.query(func.avg(AIChat.user_rating)).filter(
            AIChat.user_rating.isnot(None)
        ).scalar() or 0
        
        # Среднее время обработки
        avg_processing_time = self.db.query(func.avg(AIChat.processing_time)).scalar() or 0
        
        # Статистика по типам сообщений
        message_types = self.db.query(
            AIChat.question_type,
            func.count(AIChat.id).label('count')
        ).group_by(AIChat.question_type).all()
        
        return {
            "total_messages": total_messages,
            "total_sessions": total_sessions,
            "average_rating": float(avg_rating),
            "average_processing_time": float(avg_processing_time),
            "message_types": {mt.question_type: mt.count for mt in message_types},
            "messages_per_session": total_messages / total_sessions if total_sessions > 0 else 0
        }
    
    def delete_session(self, session_id: str, user_id: int) -> bool:
        """Удалить сессию чата"""
        deleted_count = self.db.query(AIChat).filter(
            and_(
                AIChat.session_id == session_id,
                AIChat.user_id == user_id
            )
        ).delete()
        
        self.db.commit()
        return deleted_count > 0


def get_ai_chat_crud(db: Session) -> CRUDAIChat:
    """Получить экземпляр CRUD для AI чата"""
    return CRUDAIChat(db)