from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
from datetime import datetime, timedelta

from app.models.progress import Progress, ProgressStatusEnum
from app.models.user import User
from app.models.course import Course
from app.models.lesson import Lesson
from app.schemas.progress import (
    LessonProgressCreate, LessonProgressUpdate, UserProgressSummary,
    CourseProgressWithDetails, ProgressStatus
)
class CRUDProgress:
    """CRUD операции для прогресса"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_user_progress(
        self, 
        user_id: int, 
        course_id: Optional[int] = None,
        lesson_id: Optional[int] = None
    ) -> List[Progress]:
        """Получить прогресс пользователя"""
        query = self.db.query(Progress).filter(Progress.user_id == user_id)
        
        if course_id:
            query = query.filter(Progress.course_id == course_id)
        if lesson_id:
            query = query.filter(Progress.lesson_id == lesson_id)
            
        return query.all()
    
    def get_course_progress(
        self, 
        user_id: int, 
        course_id: int
    ) -> Optional[Progress]:
        """Получить общий прогресс по курсу"""
        return self.db.query(Progress).filter(
            and_(
                Progress.user_id == user_id,
                Progress.course_id == course_id,
                Progress.lesson_id.is_(None)  # Общий прогресс по курсу
            )
        ).first()
    
    def get_lesson_progress(
        self, 
        user_id: int, 
        lesson_id: int
    ) -> Optional[Progress]:
        """Получить прогресс по уроку"""
        return self.db.query(Progress).filter(
            and_(
                Progress.user_id == user_id,
                Progress.lesson_id == lesson_id
            )
        ).first()
    
    def start_lesson(
        self, 
        user_id: int, 
        lesson_id: int
    ) -> Progress:
        """Начать урок"""
        # Получаем информацию об уроке
        lesson = self.db.query(Lesson).filter(Lesson.id == lesson_id).first()
        if not lesson:
            raise ValueError(f"Lesson {lesson_id} not found")
        
        # Проверяем существующий прогресс
        existing_progress = self.get_lesson_progress(user_id, lesson_id)
        if existing_progress:
            if existing_progress.status == ProgressStatusEnum.NOT_STARTED:
                existing_progress.start_progress()
                self.db.commit()
                self.db.refresh(existing_progress)
            return existing_progress
        
        # Создаем новый прогресс
        progress_data = {
            "user_id": user_id,
            "course_id": lesson.course_id,
            "lesson_id": lesson_id,
            "status": ProgressStatusEnum.IN_PROGRESS
        }
        progress = Progress(**progress_data)
        progress.start_progress()
        
        self.db.add(progress)
        self.db.commit()
        self.db.refresh(progress)
        
        # Обновляем общий прогресс по курсу
        self._update_course_progress(user_id, lesson.course_id)
        
        return progress
    
    def complete_lesson(
        self, 
        user_id: int, 
        lesson_id: int, 
        score: Optional[int] = None,
        max_score: Optional[int] = None
    ) -> Progress:
        """Завершить урок"""
        progress = self.get_lesson_progress(user_id, lesson_id)
        if not progress:
            raise ValueError(f"Progress for lesson {lesson_id} not found")
        
        progress.complete_progress(score, max_score)
        self.db.commit()
        self.db.refresh(progress)
        
        # Получаем информацию об уроке для обновления прогресса курса
        lesson = self.db.query(Lesson).filter(Lesson.id == lesson_id).first()
        if lesson:
            self._update_course_progress(user_id, lesson.course_id)
        
        return progress
    
    def update_lesson_time(
        self, 
        user_id: int, 
        lesson_id: int, 
        time_delta: int
    ) -> Progress:
        """Обновить время изучения урока"""
        progress = self.get_lesson_progress(user_id, lesson_id)
        if not progress:
            # Если прогресса нет, начинаем урок
            progress = self.start_lesson(user_id, lesson_id)
        
        progress.time_spent += time_delta
        self.db.commit()
        self.db.refresh(progress)
        
        return progress
    
    def get_user_progress_summary(
        self, 
        user_id: int
    ) -> UserProgressSummary:
        """Получить сводку прогресса пользователя"""
        # Общая статистика по курсам
        course_stats = self.db.query(
            func.count(Progress.course_id.distinct()).label('total_courses'),
            func.sum(
                func.case(
                    (Progress.status == ProgressStatusEnum.COMPLETED, 1),
                    else_=0
                )
            ).label('completed_courses'),
            func.sum(
                func.case(
                    (Progress.status == ProgressStatusEnum.IN_PROGRESS, 1),
                    else_=0
                )
            ).label('in_progress_courses')
        ).filter(
            and_(
                Progress.user_id == user_id,
                Progress.lesson_id.is_(None)  # Только общий прогресс по курсам
            )
        ).first()
        
        # Статистика по урокам
        lesson_stats = self.db.query(
            func.count(Progress.id).label('total_lessons'),
            func.sum(
                func.case(
                    (Progress.status == ProgressStatusEnum.COMPLETED, 1),
                    else_=0
                )
            ).label('completed_lessons'),
            func.sum(Progress.time_spent).label('total_time'),
            func.avg(Progress.score).label('avg_score')
        ).filter(
            and_(
                Progress.user_id == user_id,
                Progress.lesson_id.is_not(None)  # Только прогресс по урокам
            )
        ).first()
        
        # Последняя активность
        last_activity = self.db.query(Progress.last_accessed_at).filter(
            Progress.user_id == user_id
        ).order_by(desc(Progress.last_accessed_at)).first()
        
        return UserProgressSummary(
            user_id=user_id,
            total_courses=course_stats.total_courses or 0,
            completed_courses=course_stats.completed_courses or 0,
            in_progress_courses=course_stats.in_progress_courses or 0,
            total_lessons=lesson_stats.total_lessons or 0,
            completed_lessons=lesson_stats.completed_lessons or 0,
            total_time_spent=lesson_stats.total_time or 0,
            total_score=0,  # Будет вычислено отдельно
            max_possible_score=0,  # Будет вычислено отдельно
            overall_percentage=0.0,  # Будет вычислено отдельно
            started_at=None,  # Будет вычислено отдельно
            last_activity=last_activity.last_accessed_at if last_activity else None,
            is_course_completed=False  # Будет вычислено отдельно
        )
    
    def get_course_progress_detail(
        self, 
        user_id: int, 
        course_id: int
    ) -> Optional[CourseProgressWithDetails]:
        """Получить детальный прогресс по курсу"""
        # Получаем информацию о курсе
        course = self.db.query(Course).filter(Course.id == course_id).first()
        if not course:
            return None
        
        # Получаем все уроки курса с прогрессом
        lessons_with_progress = self.db.query(Lesson, Progress).outerjoin(
            Progress,
            and_(
                Progress.lesson_id == Lesson.id,
                Progress.user_id == user_id
            )
        ).filter(Lesson.course_id == course_id).all()
        
        # Формируем детали прогресса по урокам
        lessons_progress = []
        total_lessons = len(lessons_with_progress)
        completed_lessons = 0
        total_time = 0
        total_score = 0
        max_possible_score = 0
        
        for lesson, progress in lessons_with_progress:
            if progress:
                status = progress.status
                percentage = progress.percentage
                score = progress.score
                max_score = progress.max_score
                time_spent = progress.time_spent
                started_at = progress.started_at
                completed_at = progress.completed_at
                last_accessed = progress.last_accessed_at
                
                if status == ProgressStatusEnum.COMPLETED:
                    completed_lessons += 1
                total_time += time_spent
                if score:
                    total_score += score
                if max_score:
                    max_possible_score += max_score
            else:
                status = ProgressStatusEnum.NOT_STARTED
                percentage = 0.0
                score = None
                max_score = None
                time_spent = 0
                started_at = None
                completed_at = None
                last_accessed = datetime.now()
            
            lessons_progress.append({
                "lesson_id": lesson.id,
                "lesson_title": lesson.title,
                "lesson_type": lesson.lesson_type,
                "status": status,
                "percentage": percentage,
                "score": score,
                "max_score": max_score,
                "attempts": progress.attempts if progress else 0,
                "time_spent": time_spent,
                "time_spent_formatted": progress.time_spent_formatted if progress else "0 мин",
                "started_at": started_at,
                "completed_at": completed_at,
                "last_accessed_at": last_accessed
            })
        
        # Общий прогресс по курсу
        overall_percentage = (completed_lessons / total_lessons * 100) if total_lessons > 0 else 0
        
        overall_progress = UserProgressSummary(
            user_id=user_id,
            course_id=course_id,
            course_title=course.title,
            total_lessons=total_lessons,
            completed_lessons=completed_lessons,
            in_progress_lessons=total_lessons - completed_lessons,
            overall_percentage=overall_percentage,
            total_time_spent=total_time,
            total_score=total_score,
            max_possible_score=max_possible_score,
            started_at=None,  # Можно добавить логику для определения
            last_activity=None,  # Можно добавить логику для определения
            is_course_completed=completed_lessons == total_lessons
        )
        
        return CourseProgressDetail(
            course_id=course_id,
            course_title=course.title,
            course_description=course.description,
            overall_progress=overall_progress,
            lessons_progress=lessons_progress
        )
    
    def _update_course_progress(
        self, 
        user_id: int, 
        course_id: int
    ) -> None:
        """Обновить общий прогресс по курсу"""
        # Получаем статистику по урокам курса
        lesson_stats = self.db.query(
            func.count(Progress.id).label('total_lessons'),
            func.sum(
                func.case(
                    (Progress.status == ProgressStatusEnum.COMPLETED, 1),
                    else_=0
                )
            ).label('completed_lessons'),
            func.sum(Progress.time_spent).label('total_time'),
            func.sum(Progress.score).label('total_score'),
            func.sum(Progress.max_score).label('max_possible_score')
        ).filter(
            and_(
                Progress.user_id == user_id,
                Progress.course_id == course_id,
                Progress.lesson_id.is_not(None)
            )
        ).first()
        
        # Получаем или создаем общий прогресс по курсу
        course_progress = self.get_course_progress(user_id, course_id)
        if not course_progress:
            course_progress = Progress(
                user_id=user_id,
                course_id=course_id,
                lesson_id=None,
                status=ProgressStatusEnum.NOT_STARTED
            )
            self.db.add(course_progress)
        
        # Обновляем статистику
        total_lessons = lesson_stats.total_lessons or 0
        completed_lessons = lesson_stats.completed_lessons or 0
        
        if total_lessons > 0:
            course_progress.percentage = (completed_lessons / total_lessons) * 100
            
            if completed_lessons == 0:
                course_progress.status = ProgressStatusEnum.NOT_STARTED
            elif completed_lessons == total_lessons:
                course_progress.status = ProgressStatusEnum.COMPLETED
                if not course_progress.completed_at:
                    course_progress.completed_at = func.now()
            else:
                course_progress.status = ProgressStatusEnum.IN_PROGRESS
                if not course_progress.started_at:
                    course_progress.started_at = func.now()
        
        course_progress.time_spent = lesson_stats.total_time or 0
        course_progress.score = lesson_stats.total_score or 0
        course_progress.max_score = lesson_stats.max_possible_score or 0
        
        self.db.commit()


def get_progress_crud(db: Session) -> CRUDProgress:
    """Получить экземпляр CRUD для прогресса"""
    return CRUDProgress(db)


# Создаем экземпляр для обратной совместимости
progress = None  # Будет инициализирован при первом вызове get_progress_crud