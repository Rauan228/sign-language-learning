from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
from app.models.course import Course
from app.schemas.course import CourseCreate, CourseUpdate


class CourseCRUD:
    """CRUD операции для курсов"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get(self, course_id: int) -> Optional[Course]:
        """Получить курс по ID"""
        return self.db.query(Course).filter(Course.id == course_id).first()
    
    def get_by_teacher(self, teacher_id: int, skip: int = 0, limit: int = 100) -> List[Course]:
        """Получить курсы преподавателя"""
        return self.db.query(Course).filter(
            Course.teacher_id == teacher_id
        ).offset(skip).limit(limit).all()
    
    def get_published(self, skip: int = 0, limit: int = 100) -> List[Course]:
        """Получить опубликованные курсы"""
        return self.db.query(Course).filter(
            Course.status == "published"
        ).offset(skip).limit(limit).all()
    
    def create(self, obj_in: CourseCreate) -> Course:
        """Создать курс"""
        db_obj = Course(
            title=obj_in.title,
            description=obj_in.description,
            short_description=obj_in.short_description,
            subject=obj_in.subject,
            category=obj_in.category,
            level=obj_in.level,
            sign_language=obj_in.sign_language,
            price=obj_in.price,
            original_price=obj_in.original_price,
            is_free=obj_in.is_free,
            duration_hours=obj_in.duration_hours,
            thumbnail_url=obj_in.thumbnail_url,
            trailer_url=obj_in.trailer_url,
            teacher_id=obj_in.teacher_id
        )
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj
    
    def update(self, db_obj: Course, obj_in: CourseUpdate) -> Course:
        """Обновить курс"""
        update_data = obj_in.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj
    
    def delete(self, course_id: int) -> Optional[Course]:
        """Удалить курс"""
        course = self.get(course_id)
        if course:
            self.db.delete(course)
            self.db.commit()
        return course
    
    def count_total(self) -> int:
        """Общее количество курсов"""
        return self.db.query(Course).count()
    
    def count_published(self) -> int:
        """Количество опубликованных курсов"""
        return self.db.query(Course).filter(
            Course.status == "published"
        ).count()


def get_course_crud(db: Session) -> CourseCRUD:
    """Получить экземпляр CRUD для курсов"""
    return CourseCRUD(db)


# Для обратной совместимости
course = None  # Будет инициализирован при первом вызове get_course_crud