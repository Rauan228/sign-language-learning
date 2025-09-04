from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from datetime import datetime, timedelta

from app.models.course import Course
from app.models.user import User
from app.models.enrollment import Enrollment
from app.schemas.course import (
    CourseCreate,
    CourseUpdate,
    CourseFilter,
    CourseListResponse,
    CourseWithTeacher,
    CourseStats
)
from app.core.exceptions import (
    CourseNotFoundError,
    CourseAlreadyExistsError,
    EnrollmentError,
    PermissionDeniedError
)


class CourseService:
    def __init__(self, db: Session):
        self.db = db

    async def get_courses(
        self,
        filter_params: CourseFilter,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> CourseListResponse:
        """Получить список курсов с фильтрацией и пагинацией"""
        query = self.db.query(Course).join(User, Course.teacher_id == User.id)
        
        # Применяем фильтры
        if filter_params.level:
            query = query.filter(Course.level == filter_params.level)
        
        if filter_params.language:
            query = query.filter(Course.language == filter_params.language)
        
        if filter_params.status:
            query = query.filter(Course.status == filter_params.status)
        
        if filter_params.teacher_id:
            query = query.filter(Course.teacher_id == filter_params.teacher_id)
        
        if filter_params.is_free is not None:
            if filter_params.is_free:
                query = query.filter(Course.price == 0)
            else:
                query = query.filter(Course.price > 0)
        
        if filter_params.search:
            search_term = f"%{filter_params.search}%"
            query = query.filter(
                or_(
                    Course.title.ilike(search_term),
                    Course.description.ilike(search_term),
                    Course.short_description.ilike(search_term)
                )
            )
        
        # Показываем только опубликованные курсы для обычных пользователей
        if not filter_params.include_unpublished:
            query = query.filter(Course.is_published == True)
        
        # Подсчитываем общее количество
        total = query.count()
        
        # Применяем сортировку
        if hasattr(Course, sort_by):
            order_func = desc if sort_order == "desc" else asc
            query = query.order_by(order_func(getattr(Course, sort_by)))
        
        # Применяем пагинацию
        offset = (page - 1) * page_size
        courses = query.offset(offset).limit(page_size).all()
        
        # Преобразуем в схемы с информацией о преподавателе
        courses_with_teacher = []
        for course in courses:
            course_dict = {
                **course.__dict__,
                "teacher_name": course.teacher.full_name,
                "teacher_avatar": course.teacher.avatar_url,
                "teacher_rating": course.teacher.rating or 0.0
            }
            courses_with_teacher.append(CourseWithTeacher(**course_dict))
        
        total_pages = (total + page_size - 1) // page_size
        
        return CourseListResponse(
            courses=courses_with_teacher,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    async def get_course_by_id(self, course_id: int) -> Optional[CourseWithTeacher]:
        """Получить курс по ID"""
        course = self.db.query(Course).join(User, Course.teacher_id == User.id).filter(Course.id == course_id).first()
        
        if not course:
            return None
        
        course_dict = {
            **course.__dict__,
            "teacher_name": course.teacher.full_name,
            "teacher_avatar": course.teacher.avatar_url,
            "teacher_rating": course.teacher.rating or 0.0
        }
        
        return CourseWithTeacher(**course_dict)

    async def create_course(self, course_data: CourseCreate) -> Course:
        """Создать новый курс"""
        # Проверяем, что преподаватель существует
        teacher = self.db.query(User).filter(
            and_(
                User.id == course_data.teacher_id,
                User.role.in_(["teacher", "admin"])
            )
        ).first()
        
        if not teacher:
            raise CourseNotFoundError("Преподаватель не найден")
        
        # Проверяем уникальность названия курса у преподавателя
        existing_course = self.db.query(Course).filter(
            and_(
                Course.title == course_data.title,
                Course.teacher_id == course_data.teacher_id
            )
        ).first()
        
        if existing_course:
            raise CourseAlreadyExistsError("Курс с таким названием уже существует у данного преподавателя")
        
        # Создаем курс
        db_course = Course(**course_data.dict())
        self.db.add(db_course)
        self.db.commit()
        self.db.refresh(db_course)
        
        return db_course

    async def update_course(self, course_id: int, course_data: CourseUpdate) -> Course:
        """Обновить курс"""
        course = self.db.query(Course).filter(Course.id == course_id).first()
        
        if not course:
            raise CourseNotFoundError("Курс не найден")
        
        # Обновляем только переданные поля
        update_data = course_data.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            setattr(course, field, value)
        
        course.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(course)
        
        return course

    async def delete_course(self, course_id: int) -> bool:
        """Удалить курс"""
        course = self.db.query(Course).filter(Course.id == course_id).first()
        
        if not course:
            raise CourseNotFoundError("Курс не найден")
        
        # Проверяем, есть ли записанные студенты
        enrolled_count = self.db.query(Enrollment).filter(Enrollment.course_id == course_id).count()
        
        if enrolled_count > 0:
            raise EnrollmentError("Нельзя удалить курс с записанными студентами")
        
        self.db.delete(course)
        self.db.commit()
        
        return True

    async def enroll_user(self, course_id: int, user_id: int) -> Enrollment:
        """Записать пользователя на курс"""
        # Проверяем существование курса
        course = self.db.query(Course).filter(Course.id == course_id).first()
        if not course:
            raise CourseNotFoundError("Курс не найден")
        
        # Проверяем, что курс опубликован
        if not course.is_published:
            raise EnrollmentError("Нельзя записаться на неопубликованный курс")
        
        # Проверяем, что пользователь не записан на курс
        existing_enrollment = self.db.query(Enrollment).filter(
            and_(
                Enrollment.course_id == course_id,
                Enrollment.user_id == user_id
            )
        ).first()
        
        if existing_enrollment:
            raise EnrollmentError("Пользователь уже записан на этот курс")
        
        # Создаем запись
        enrollment = Enrollment(
            course_id=course_id,
            user_id=user_id,
            enrolled_at=datetime.utcnow()
        )
        
        self.db.add(enrollment)
        
        # Увеличиваем счетчик студентов
        course.students_count += 1
        
        self.db.commit()
        self.db.refresh(enrollment)
        
        return enrollment

    async def unenroll_user(self, course_id: int, user_id: int) -> bool:
        """Отписать пользователя от курса"""
        enrollment = self.db.query(Enrollment).filter(
            and_(
                Enrollment.course_id == course_id,
                Enrollment.user_id == user_id
            )
        ).first()
        
        if not enrollment:
            raise EnrollmentError("Пользователь не записан на этот курс")
        
        # Удаляем запись
        self.db.delete(enrollment)
        
        # Уменьшаем счетчик студентов
        course = self.db.query(Course).filter(Course.id == course_id).first()
        if course and course.students_count > 0:
            course.students_count -= 1
        
        self.db.commit()
        
        return True

    async def get_enrolled_courses(
        self,
        user_id: int,
        page: int = 1,
        page_size: int = 20
    ) -> CourseListResponse:
        """Получить курсы, на которые записан пользователь"""
        query = self.db.query(Course).join(
            Enrollment, Course.id == Enrollment.course_id
        ).join(
            User, Course.teacher_id == User.id
        ).filter(Enrollment.user_id == user_id)
        
        total = query.count()
        
        offset = (page - 1) * page_size
        courses = query.offset(offset).limit(page_size).all()
        
        courses_with_teacher = []
        for course in courses:
            course_dict = {
                **course.__dict__,
                "teacher_name": course.teacher.full_name,
                "teacher_avatar": course.teacher.avatar_url,
                "teacher_rating": course.teacher.rating or 0.0
            }
            courses_with_teacher.append(CourseWithTeacher(**course_dict))
        
        total_pages = (total + page_size - 1) // page_size
        
        return CourseListResponse(
            courses=courses_with_teacher,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    async def get_featured_courses(self, limit: int = 6) -> List[CourseWithTeacher]:
        """Получить рекомендуемые курсы"""
        courses = self.db.query(Course).join(
            User, Course.teacher_id == User.id
        ).filter(
            and_(
                Course.is_published == True,
                Course.is_featured == True
            )
        ).order_by(desc(Course.rating)).limit(limit).all()
        
        courses_with_teacher = []
        for course in courses:
            course_dict = {
                **course.__dict__,
                "teacher_name": course.teacher.full_name,
                "teacher_avatar": course.teacher.avatar_url,
                "teacher_rating": course.teacher.rating or 0.0
            }
            courses_with_teacher.append(CourseWithTeacher(**course_dict))
        
        return courses_with_teacher

    async def get_popular_courses(self, limit: int = 6) -> List[CourseWithTeacher]:
        """Получить популярные курсы"""
        courses = self.db.query(Course).join(
            User, Course.teacher_id == User.id
        ).filter(
            Course.is_published == True
        ).order_by(
            desc(Course.students_count),
            desc(Course.rating)
        ).limit(limit).all()
        
        courses_with_teacher = []
        for course in courses:
            course_dict = {
                **course.__dict__,
                "teacher_name": course.teacher.full_name,
                "teacher_avatar": course.teacher.avatar_url,
                "teacher_rating": course.teacher.rating or 0.0
            }
            courses_with_teacher.append(CourseWithTeacher(**course_dict))
        
        return courses_with_teacher

    async def publish_course(self, course_id: int) -> Course:
        """Опубликовать курс"""
        course = self.db.query(Course).filter(Course.id == course_id).first()
        
        if not course:
            raise CourseNotFoundError("Курс не найден")
        
        course.is_published = True
        course.published_at = datetime.utcnow()
        course.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(course)
        
        return course

    async def unpublish_course(self, course_id: int) -> Course:
        """Снять курс с публикации"""
        course = self.db.query(Course).filter(Course.id == course_id).first()
        
        if not course:
            raise CourseNotFoundError("Курс не найден")
        
        course.is_published = False
        course.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(course)
        
        return course

    async def get_course_students(
        self,
        course_id: int,
        page: int = 1,
        page_size: int = 20
    ) -> List[Dict[str, Any]]:
        """Получить список студентов курса"""
        query = self.db.query(User).join(
            Enrollment, User.id == Enrollment.user_id
        ).filter(Enrollment.course_id == course_id)
        
        offset = (page - 1) * page_size
        students = query.offset(offset).limit(page_size).all()
        
        students_data = []
        for student in students:
            enrollment = self.db.query(Enrollment).filter(
                and_(
                    Enrollment.course_id == course_id,
                    Enrollment.user_id == student.id
                )
            ).first()
            
            students_data.append({
                "id": student.id,
                "full_name": student.full_name,
                "email": student.email,
                "avatar_url": student.avatar_url,
                "enrolled_at": enrollment.enrolled_at,
                "progress": 0  # TODO: Добавить расчет прогресса
            })
        
        return students_data

    async def get_course_stats(self) -> CourseStats:
        """Получить общую статистику курсов"""
        total_courses = self.db.query(Course).count()
        published_courses = self.db.query(Course).filter(Course.is_published == True).count()
        total_students = self.db.query(Enrollment).count()
        
        # Средний рейтинг курсов
        avg_rating = self.db.query(func.avg(Course.rating)).scalar() or 0.0
        
        # Курсы по уровням
        level_stats = self.db.query(
            Course.level,
            func.count(Course.id)
        ).group_by(Course.level).all()
        
        # Курсы по языкам
        language_stats = self.db.query(
            Course.language,
            func.count(Course.id)
        ).group_by(Course.language).all()
        
        return CourseStats(
            total_courses=total_courses,
            published_courses=published_courses,
            total_students=total_students,
            average_rating=round(avg_rating, 2),
            courses_by_level=dict(level_stats),
            courses_by_language=dict(language_stats)
        )

    async def get_course_detailed_stats(self, course_id: int) -> Dict[str, Any]:
        """Получить детальную статистику курса"""
        course = self.db.query(Course).filter(Course.id == course_id).first()
        
        if not course:
            raise CourseNotFoundError("Курс не найден")
        
        # Количество записанных студентов
        total_students = self.db.query(Enrollment).filter(Enrollment.course_id == course_id).count()
        
        # Записи за последние 30 дней
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_enrollments = self.db.query(Enrollment).filter(
            and_(
                Enrollment.course_id == course_id,
                Enrollment.enrolled_at >= thirty_days_ago
            )
        ).count()
        
        return {
            "course_id": course_id,
            "title": course.title,
            "total_students": total_students,
            "recent_enrollments": recent_enrollments,
            "rating": course.rating,
            "reviews_count": 0,  # TODO: Добавить подсчет отзывов
            "completion_rate": 0.0,  # TODO: Добавить расчет процента завершения
            "average_progress": 0.0,  # TODO: Добавить средний прогресс
            "created_at": course.created_at,
            "published_at": course.published_at,
            "is_published": course.is_published
        }


def get_course_service(db: Session) -> CourseService:
    """Получить экземпляр сервиса курсов"""
    return CourseService(db)