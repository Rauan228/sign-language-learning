from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.crud.lesson import get_lesson_crud
from app.services.course import get_course_service
from app.models.user import User
from app.models.lesson import Lesson
from app.schemas.lesson import (
    LessonCreate,
    LessonUpdate,
    QuizSubmission,
    QuizResult,
    LessonProgress
)


class LessonService:
    """Сервис для работы с уроками"""
    
    def __init__(self, db: Session):
        self.db = db
        self.lesson_crud = get_lesson_crud(db)
        self.course_service = get_course_service(db)
    
    def validate_lesson_access(
        self, 
        db: Session, 
        lesson: Lesson, 
        user: Optional[User] = None,
        require_published: bool = True
    ) -> bool:
        """
        Проверить доступ пользователя к уроку
        """
        # Если урок не опубликован
        if require_published and not lesson.is_published:
            if not user:
                return False
            
            # Только автор курса, админ или преподаватель могут видеть неопубликованные уроки
            if (
                user.role == "student" and 
                lesson.course.teacher_id != user.id
            ):
                return False
        
        # Если урок платный, проверяем подписку/покупку
        if not lesson.is_free and user:
            # TODO: Добавить проверку подписки/покупки курса
            pass
        
        return True
    
    def get_lesson_with_access_check(
        self, 
        db: Session, 
        lesson_id: int, 
        user: Optional[User] = None
    ) -> Lesson:
        """
        Получить урок с проверкой доступа
        """
        lesson = self.lesson_crud.get(lesson_id=lesson_id)
        if not lesson:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Урок не найден"
            )
        
        if not self.validate_lesson_access(db, lesson, user):
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Требуется авторизация для доступа к этому уроку"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Недостаточно прав для доступа к этому уроку"
                )
        
        return lesson
    
    def create_lesson_with_validation(
        self, 
        db: Session, 
        lesson_in: LessonCreate, 
        user: User
    ) -> Lesson:
        """
        Создать урок с валидацией
        """
        # Проверяем существование курса
        course = self.course_service.get_course_by_id(lesson_in.course_id)
        if not course:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Курс не найден"
            )
        
        # Проверяем права на курс
        if user.role == "teacher" and course.teacher_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Недостаточно прав для создания урока в этом курсе"
            )
        
        # Валидируем данные урока
        self._validate_lesson_data(lesson_in)
        
        # Устанавливаем порядковый номер, если не указан
        if not lesson_in.order_num:
            lesson_in.order_num = self.lesson_crud.get_next_order_num(
                db=db, course_id=lesson_in.course_id
            )
        
        return self.lesson_crud.create(obj_in=lesson_in)
    
    def _validate_lesson_data(self, lesson_data: LessonCreate) -> None:
        """
        Валидировать данные урока
        """
        # Проверяем обязательные поля в зависимости от типа урока
        if lesson_data.lesson_type == "video":
            if not lesson_data.video_url:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Для видео урока обязательно указать URL видео"
                )
        
        elif lesson_data.lesson_type == "theory":
            if not lesson_data.theory_content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Для теоретического урока обязательно указать содержание"
                )
        
        elif lesson_data.lesson_type == "practice":
            if not lesson_data.practice_tasks:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Для практического урока обязательно указать задания"
                )
        
        elif lesson_data.lesson_type == "quiz":
            if not lesson_data.quiz_questions:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Для урока-теста обязательно указать вопросы"
                )
            
            # Валидируем структуру вопросов
            self._validate_quiz_questions(lesson_data.quiz_questions)
    
    def _validate_quiz_questions(self, quiz_questions: List[Dict[str, Any]]) -> None:
        """
        Валидировать структуру вопросов теста
        """
        if not quiz_questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Тест должен содержать хотя бы один вопрос"
            )
        
        for i, question in enumerate(quiz_questions):
            if not isinstance(question, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Вопрос {i+1} должен быть объектом"
                )
            
            required_fields = ["question", "options", "correct_answer"]
            for field in required_fields:
                if field not in question:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Вопрос {i+1}: отсутствует поле '{field}'"
                    )
            
            # Проверяем опции
            options = question.get("options", [])
            if not isinstance(options, list) or len(options) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Вопрос {i+1}: должно быть минимум 2 варианта ответа"
                )
            
            # Проверяем правильный ответ
            correct_answer = question.get("correct_answer")
            if correct_answer not in range(len(options)):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Вопрос {i+1}: неверный индекс правильного ответа"
                )
    
    def submit_quiz(
        self, 
        db: Session, 
        lesson_id: int, 
        submission: QuizSubmission, 
        user: User
    ) -> QuizResult:
        """
        Отправить ответы на тест
        """
        lesson = self.get_lesson_with_access_check(db, lesson_id, user)
        
        if lesson.lesson_type != "quiz":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Этот урок не является тестом"
            )
        
        if not lesson.quiz_questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="В уроке нет вопросов для теста"
            )
        
        # Проверяем количество ответов
        if len(submission.answers) != len(lesson.quiz_questions):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Количество ответов не соответствует количеству вопросов"
            )
        
        # Подсчитываем результат
        correct_answers = 0
        total_questions = len(lesson.quiz_questions)
        
        for i, (user_answer, question) in enumerate(zip(submission.answers, lesson.quiz_questions)):
            if user_answer == question.get("correct_answer"):
                correct_answers += 1
        
        score = (correct_answers / total_questions) * 100
        passed = score >= (lesson.min_score or 70)  # По умолчанию 70%
        
        # TODO: Сохранить результат в базе данных (таблица progress или quiz_results)
        
        return QuizResult(
            lesson_id=lesson_id,
            user_id=user.id,
            score=score,
            correct_answers=correct_answers,
            total_questions=total_questions,
            passed=passed,
            answers=submission.answers
        )
    
    def get_lesson_progress(
        self, 
        db: Session, 
        lesson_id: int, 
        user: User
    ) -> Optional[LessonProgress]:
        """
        Получить прогресс пользователя по уроку
        """
        lesson = self.get_lesson_with_access_check(db, lesson_id, user)
        
        # TODO: Получить прогресс из базы данных
        # Пока возвращаем заглушку
        return LessonProgress(
            lesson_id=lesson_id,
            user_id=user.id,
            completed=False,
            progress_percentage=0,
            time_spent=0,
            last_accessed=None,
            quiz_attempts=0,
            best_quiz_score=None
        )
    
    def mark_lesson_completed(
        self, 
        db: Session, 
        lesson_id: int, 
        user: User
    ) -> LessonProgress:
        """
        Отметить урок как завершенный
        """
        lesson = self.get_lesson_with_access_check(db, lesson_id, user)
        
        # TODO: Обновить прогресс в базе данных
        # Пока возвращаем заглушку
        return LessonProgress(
            lesson_id=lesson_id,
            user_id=user.id,
            completed=True,
            progress_percentage=100,
            time_spent=0,
            last_accessed=None,
            quiz_attempts=0,
            best_quiz_score=None
        )
    
    def get_course_lessons_with_progress(
        self, 
        db: Session, 
        course_id: int, 
        user: Optional[User] = None
    ) -> List[Dict[str, Any]]:
        """
        Получить уроки курса с информацией о прогрессе
        """
        # Проверяем существование курса
        course = self.course_service.get_course_by_id(course_id)
        if not course:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Курс не найден"
            )
        
        # Получаем уроки
        if not user or (
            user.role == "student" and 
            course.teacher_id != user.id
        ):
            lessons = self.lesson_crud.get_published_by_course(course_id=course_id)
        else:
            lessons = self.lesson_crud.get_by_course(course_id=course_id)
        
        # Добавляем информацию о прогрессе
        lessons_with_progress = []
        for lesson in lessons:
            lesson_data = {
                "lesson": lesson,
                "progress": None,
                "accessible": True
            }
            
            if user:
                # TODO: Получить реальный прогресс из базы данных
                lesson_data["progress"] = self.get_lesson_progress(db, lesson.id, user)
                
                # Проверяем доступность урока
                lesson_data["accessible"] = self.validate_lesson_access(
                    db, lesson, user, require_published=False
                )
            else:
                lesson_data["accessible"] = lesson.is_published and lesson.is_free
            
            lessons_with_progress.append(lesson_data)
        
        return lessons_with_progress


def get_lesson_service(db: Session) -> LessonService:
    """Получить экземпляр сервиса уроков"""
    return LessonService(db)