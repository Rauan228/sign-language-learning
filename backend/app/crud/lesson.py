from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from app.models.lesson import Lesson
from app.schemas.lesson import LessonCreate, LessonUpdate, LessonFilter


class LessonCRUD:
    """CRUD операции для уроков"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, obj_in: LessonCreate) -> Lesson:
        """Создать урок"""
        db_obj = Lesson(
            course_id=obj_in.course_id,
            title=obj_in.title,
            description=obj_in.description,
            video_url=obj_in.video_url,
            video_duration=obj_in.video_duration,
            materials=obj_in.materials,
            theory_content=obj_in.theory_content,
            practice_tasks=obj_in.practice_tasks,
            quiz_questions=obj_in.quiz_questions,
            order_num=obj_in.order_num,
            lesson_type=obj_in.lesson_type,
            is_free=obj_in.is_free,
            is_published=obj_in.is_published,
            min_score=obj_in.min_score,
            max_attempts=obj_in.max_attempts
        )
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj
    
    def get(self, lesson_id: int) -> Optional[Lesson]:
        """Получить урок по ID"""
        return self.db.query(Lesson).filter(Lesson.id == lesson_id).first()
    
    def get_by_course(self, db: Session, course_id: int, skip: int = 0, limit: int = 100) -> List[Lesson]:
        """Получить уроки курса"""
        return (
            db.query(Lesson)
            .filter(Lesson.course_id == course_id)
            .order_by(Lesson.order_num)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_published_by_course(self, db: Session, course_id: int) -> List[Lesson]:
        """Получить опубликованные уроки курса"""
        return (
            db.query(Lesson)
            .filter(
                and_(
                    Lesson.course_id == course_id,
                    Lesson.is_published == True
                )
            )
            .order_by(Lesson.order_num)
            .all()
        )
    
    def get_multi_with_filter(
        self, 
        db: Session, 
        *,
        filter_params: LessonFilter,
        skip: int = 0, 
        limit: int = 100
    ) -> List[Lesson]:
        """Получить уроки с фильтрацией"""
        query = db.query(Lesson)
        
        # Фильтр по курсу
        if filter_params.course_id:
            query = query.filter(Lesson.course_id == filter_params.course_id)
        
        # Фильтр по типу урока
        if filter_params.lesson_type:
            query = query.filter(Lesson.lesson_type == filter_params.lesson_type)
        
        # Фильтр по бесплатности
        if filter_params.is_free is not None:
            query = query.filter(Lesson.is_free == filter_params.is_free)
        
        # Фильтр по публикации
        if filter_params.is_published is not None:
            query = query.filter(Lesson.is_published == filter_params.is_published)
        
        # Поиск по названию и описанию
        if filter_params.search:
            search_term = f"%{filter_params.search}%"
            query = query.filter(
                or_(
                    Lesson.title.ilike(search_term),
                    Lesson.description.ilike(search_term)
                )
            )
        
        return (
            query
            .order_by(Lesson.course_id, Lesson.order_num)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def count_with_filter(self, db: Session, *, filter_params: LessonFilter) -> int:
        """Подсчитать количество уроков с фильтрацией"""
        query = db.query(Lesson)
        
        if filter_params.course_id:
            query = query.filter(Lesson.course_id == filter_params.course_id)
        
        if filter_params.lesson_type:
            query = query.filter(Lesson.lesson_type == filter_params.lesson_type)
        
        if filter_params.is_free is not None:
            query = query.filter(Lesson.is_free == filter_params.is_free)
        
        if filter_params.is_published is not None:
            query = query.filter(Lesson.is_published == filter_params.is_published)
        
        if filter_params.search:
            search_term = f"%{filter_params.search}%"
            query = query.filter(
                or_(
                    Lesson.title.ilike(search_term),
                    Lesson.description.ilike(search_term)
                )
            )
        
        return query.count()
    
    def update(self, db: Session, *, db_obj: Lesson, obj_in: LessonUpdate) -> Lesson:
        """Обновить урок"""
        update_data = obj_in.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def delete(self, db: Session, *, lesson_id: int) -> Optional[Lesson]:
        """Удалить урок"""
        obj = db.query(Lesson).filter(Lesson.id == lesson_id).first()
        if obj:
            db.delete(obj)
            db.commit()
        return obj
    
    def reorder_lessons(self, db: Session, course_id: int, lesson_orders: Dict[int, int]) -> bool:
        """Изменить порядок уроков в курсе"""
        try:
            for lesson_id, new_order in lesson_orders.items():
                lesson = db.query(Lesson).filter(
                    and_(
                        Lesson.id == lesson_id,
                        Lesson.course_id == course_id
                    )
                ).first()
                
                if lesson:
                    lesson.order_num = new_order
                    db.add(lesson)
            
            db.commit()
            return True
        except Exception:
            db.rollback()
            return False
    
    def get_next_order_num(self, db: Session, course_id: int) -> int:
        """Получить следующий порядковый номер для урока в курсе"""
        max_order = (
            db.query(Lesson.order_num)
            .filter(Lesson.course_id == course_id)
            .order_by(Lesson.order_num.desc())
            .first()
        )
        
        return (max_order[0] + 1) if max_order else 1
    
    def duplicate_lesson(self, db: Session, lesson_id: int, new_course_id: Optional[int] = None) -> Optional[Lesson]:
        """Дублировать урок"""
        original = self.get(db, lesson_id)
        if not original:
            return None
        
        target_course_id = new_course_id or original.course_id
        new_order = self.get_next_order_num(db, target_course_id)
        
        new_lesson = Lesson(
            course_id=target_course_id,
            title=f"{original.title} (копия)",
            description=original.description,
            video_url=original.video_url,
            video_duration=original.video_duration,
            materials=original.materials,
            theory_content=original.theory_content,
            practice_tasks=original.practice_tasks,
            quiz_questions=original.quiz_questions,
            order_num=new_order,
            lesson_type=original.lesson_type,
            is_free=original.is_free,
            is_published=False,  # Копия не публикуется автоматически
            min_score=original.min_score,
            max_attempts=original.max_attempts
        )
        
        db.add(new_lesson)
        db.commit()
        db.refresh(new_lesson)
        return new_lesson


def get_lesson_crud(db: Session) -> LessonCRUD:
    """Получить экземпляр CRUD для уроков"""
    return LessonCRUD(db)