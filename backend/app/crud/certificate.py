from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc, extract
from datetime import datetime, timedelta
import secrets
import string

from app.models.certificate import Certificate
from app.models.user import User
from app.models.course import Course
from app.models.progress import Progress, ProgressStatusEnum
from app.schemas.certificate import (
    CertificateCreate, CertificateUpdate, CertificateWithDetails, CertificateStats
)


class CRUDCertificate:
    """CRUD операции для сертификатов"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def _generate_certificate_number(self) -> str:
        """Генерация уникального номера сертификата"""
        while True:
            # Формат: CERT-YYYY-XXXXXXXX (где X - случайные символы)
            year = datetime.now().year
            random_part = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            certificate_number = f"CERT-{year}-{random_part}"
            
            # Проверяем уникальность
            existing = self.db.query(Certificate).filter(
                Certificate.certificate_number == certificate_number
            ).first()
            
            if not existing:
                return certificate_number
    
    def create(self, obj_in: CertificateCreate) -> Certificate:
        """Создать сертификат"""
        # Проверяем, что пользователь завершил курс
        course_progress = self.db.query(Progress).filter(
            and_(
                Progress.user_id == obj_in.user_id,
                Progress.course_id == obj_in.course_id,
                Progress.lesson_id.is_(None),  # Общий прогресс по курсу
                Progress.status == ProgressStatusEnum.COMPLETED
            )
        ).first()
        
        if not course_progress:
            raise ValueError("Пользователь не завершил курс")
        
        # Проверяем, что сертификат еще не выдан
        existing_certificate = self.db.query(Certificate).filter(
            and_(
                Certificate.user_id == obj_in.user_id,
                Certificate.course_id == obj_in.course_id,
                Certificate.is_active == True
            )
        ).first()
        
        if existing_certificate:
            raise ValueError("Сертификат уже выдан для этого курса")
        
        # Получаем информацию о курсе
        course = self.db.query(Course).filter(Course.id == obj_in.course_id).first()
        if not course:
            raise ValueError("Курс не найден")
        
        # Создаем сертификат
        certificate_data = {
            "user_id": obj_in.user_id,
            "course_id": obj_in.course_id,
            "certificate_number": self._generate_certificate_number(),
            "title": obj_in.title or f"Сертификат о прохождении курса '{course.title}'",
            "description": obj_in.description,
            "completion_date": course_progress.completed_at or datetime.utcnow(),
            "total_score": obj_in.total_score or course_progress.score or 0,
            "max_possible_score": obj_in.max_possible_score or course_progress.max_score or 0,
            "percentage": obj_in.percentage or int(course_progress.percentage or 0),
            "total_time_spent": obj_in.total_time_spent or course_progress.time_spent or 0
        }
        
        certificate = Certificate(**certificate_data)
        self.db.add(certificate)
        self.db.commit()
        self.db.refresh(certificate)
        
        return certificate
    
    def get(self, certificate_id: int) -> Optional[Certificate]:
        """Получить сертификат по ID"""
        return self.db.query(Certificate).filter(Certificate.id == certificate_id).first()
    
    def get_by_number(self, certificate_number: str) -> Optional[Certificate]:
        """Получить сертификат по номеру"""
        return self.db.query(Certificate).filter(
            Certificate.certificate_number == certificate_number
        ).first()
    
    def get_user_certificates(
        self, 
        user_id: int, 
        skip: int = 0, 
        limit: int = 100,
        active_only: bool = True
    ) -> List[Certificate]:
        """Получить сертификаты пользователя"""
        query = self.db.query(Certificate).filter(Certificate.user_id == user_id)
        
        if active_only:
            query = query.filter(Certificate.is_active == True)
        
        return query.order_by(desc(Certificate.created_at)).offset(skip).limit(limit).all()
    
    def get_course_certificates(
        self, 
        course_id: int, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[Certificate]:
        """Получить сертификаты по курсу"""
        return self.db.query(Certificate).filter(
            and_(
                Certificate.course_id == course_id,
                Certificate.is_active == True
            )
        ).order_by(desc(Certificate.created_at)).offset(skip).limit(limit).all()
    
    def get_with_details(self, certificate_id: int) -> Optional[CertificateWithDetails]:
        """Получить сертификат с подробной информацией"""
        result = self.db.query(
            Certificate,
            User.full_name.label('user_name'),
            User.email.label('user_email'),
            Course.title.label('course_title'),
            Course.description.label('course_description')
        ).join(
            User, Certificate.user_id == User.id
        ).join(
            Course, Certificate.course_id == Course.id
        ).filter(Certificate.id == certificate_id).first()
        
        if not result:
            return None
        
        certificate, user_name, user_email, course_title, course_description = result
        
        return CertificateWithDetails(
            **certificate.__dict__,
            user_name=user_name,
            user_email=user_email,
            course_title=course_title,
            course_description=course_description
        )
    
    def update(self, certificate_id: int, obj_in: CertificateUpdate) -> Optional[Certificate]:
        """Обновить сертификат"""
        certificate = self.get(certificate_id)
        if not certificate:
            return None
        
        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(certificate, field, value)
        
        certificate.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(certificate)
        
        return certificate
    
    def deactivate(self, certificate_id: int) -> Optional[Certificate]:
        """Деактивировать сертификат"""
        certificate = self.get(certificate_id)
        if not certificate:
            return None
        
        certificate.is_active = False
        certificate.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(certificate)
        
        return certificate
    
    def get_stats(self) -> CertificateStats:
        """Получить статистику по сертификатам"""
        # Общее количество сертификатов
        total_certificates = self.db.query(func.count(Certificate.id)).scalar() or 0
        
        # Активные сертификаты
        active_certificates = self.db.query(func.count(Certificate.id)).filter(
            Certificate.is_active == True
        ).scalar() or 0
        
        # Сертификаты за текущий месяц
        current_month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        certificates_this_month = self.db.query(func.count(Certificate.id)).filter(
            and_(
                Certificate.created_at >= current_month_start,
                Certificate.is_active == True
            )
        ).scalar() or 0
        
        # Средний балл
        avg_score_result = self.db.query(func.avg(Certificate.percentage)).filter(
            Certificate.is_active == True
        ).scalar()
        average_score = float(avg_score_result) if avg_score_result else 0.0
        
        # Среднее время прохождения
        avg_time_result = self.db.query(func.avg(Certificate.total_time_spent)).filter(
            Certificate.is_active == True
        ).scalar()
        average_completion_time = int(avg_time_result) if avg_time_result else 0
        
        return CertificateStats(
            total_certificates=total_certificates,
            active_certificates=active_certificates,
            certificates_this_month=certificates_this_month,
            average_score=average_score,
            average_completion_time=average_completion_time
        )
    
    def can_issue_certificate(self, user_id: int, course_id: int) -> bool:
        """Проверить, можно ли выдать сертификат"""
        # Проверяем завершение курса
        course_progress = self.db.query(Progress).filter(
            and_(
                Progress.user_id == user_id,
                Progress.course_id == course_id,
                Progress.lesson_id.is_(None),
                Progress.status == ProgressStatusEnum.COMPLETED
            )
        ).first()
        
        if not course_progress:
            return False
        
        # Проверяем, что сертификат еще не выдан
        existing_certificate = self.db.query(Certificate).filter(
            and_(
                Certificate.user_id == user_id,
                Certificate.course_id == course_id,
                Certificate.is_active == True
            )
        ).first()
        
        return existing_certificate is None


def get_certificate_crud(db: Session) -> CRUDCertificate:
    """Получить экземпляр CRUD для сертификатов"""
    return CRUDCertificate(db)