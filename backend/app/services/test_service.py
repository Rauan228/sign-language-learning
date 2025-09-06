from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.crud.test_crud import TestCRUD
from app.schemas.test import (
    Test, TestCreate, TestUpdate, TestAttempt, TestAttemptCreate,
    TestAttemptSubmit, TestResult, TestStatistics, UserTestProgress
)
from app.models.user import User
from app.models.lesson import Lesson


class TestService:
    def __init__(self, db: Session):
        self.db = db
        self.test_crud = TestCRUD(db)

    def create_test(self, test_data: TestCreate, current_user: User) -> Test:
        """Создать новый тест"""
        # Проверяем, что урок существует
        lesson = self.db.query(Lesson).filter(Lesson.id == test_data.lesson_id).first()
        if not lesson:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Урок не найден"
            )

        # Проверяем права доступа (только преподаватели и админы могут создавать тесты)
        if current_user.role not in ["teacher", "admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Недостаточно прав для создания теста"
            )

        # Валидируем данные теста
        self._validate_test_data(test_data)

        try:
            test = self.test_crud.create_test(test_data)
            return test
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ошибка при создании теста: {str(e)}"
            )

    def get_test(self, test_id: int, current_user: User) -> Test:
        """Получить тест по ID"""
        test = self.test_crud.get_test(test_id)
        if not test:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Тест не найден"
            )

        # Проверяем права доступа
        if not self._can_access_test(test, current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Недостаточно прав для просмотра теста"
            )

        return test

    def get_tests_by_lesson(self, lesson_id: int, current_user: User) -> List[Test]:
        """Получить все тесты урока"""
        # Проверяем, что урок существует
        lesson = self.db.query(Lesson).filter(Lesson.id == lesson_id).first()
        if not lesson:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Урок не найден"
            )

        # Для студентов показываем только активные тесты
        active_only = current_user.role == "student"
        tests = self.test_crud.get_tests_by_lesson(lesson_id, active_only)
        
        return tests

    def update_test(self, test_id: int, test_data: TestUpdate, current_user: User) -> Test:
        """Обновить тест"""
        test = self.test_crud.get_test(test_id)
        if not test:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Тест не найден"
            )

        # Проверяем права доступа
        if current_user.role not in ["teacher", "admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Недостаточно прав для редактирования теста"
            )

        # Валидируем данные если переданы вопросы
        if test_data.questions is not None:
            self._validate_questions(test_data.questions)

        try:
            updated_test = self.test_crud.update_test(test_id, test_data)
            return updated_test
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ошибка при обновлении теста: {str(e)}"
            )

    def delete_test(self, test_id: int, current_user: User) -> bool:
        """Удалить тест"""
        test = self.test_crud.get_test(test_id)
        if not test:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Тест не найден"
            )

        # Проверяем права доступа
        if current_user.role not in ["teacher", "admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Недостаточно прав для удаления теста"
            )

        try:
            return self.test_crud.delete_test(test_id)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ошибка при удалении теста: {str(e)}"
            )

    def start_test_attempt(self, test_id: int, current_user: User) -> TestAttempt:
        """Начать прохождение теста"""
        test = self.test_crud.get_test(test_id)
        if not test:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Тест не найден"
            )

        if not test.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Тест неактивен"
            )

        # Проверяем права доступа
        if not self._can_access_test(test, current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Недостаточно прав для прохождения теста"
            )

        # Проверяем, есть ли незавершенная попытка
        existing_attempts = self.test_crud.get_user_attempts(current_user.id, test_id)
        incomplete_attempt = next((a for a in existing_attempts if not a.is_completed), None)
        
        if incomplete_attempt:
            return incomplete_attempt

        # Создаем новую попытку
        attempt = self.test_crud.create_attempt(current_user.id, test_id)
        if not attempt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Превышено максимальное количество попыток"
            )

        return attempt

    def submit_test_attempt(self, attempt_id: int, submission: TestAttemptSubmit, current_user: User) -> TestResult:
        """Завершить прохождение теста"""
        attempt = self.test_crud.get_attempt(attempt_id)
        if not attempt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Попытка не найдена"
            )

        if attempt.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Недостаточно прав для завершения попытки"
            )

        if attempt.is_completed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Попытка уже завершена"
            )

        try:
            completed_attempt = self.test_crud.submit_attempt(attempt_id, submission)
            if not completed_attempt:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Ошибка при завершении попытки"
                )

            test = self.test_crud.get_test(completed_attempt.test_id)
            is_passed = completed_attempt.score >= test.passing_score
            
            # Подсчитываем количество правильных ответов
            correct_answers = len([a for a in completed_attempt.answers if a.is_correct])
            total_questions = len(completed_attempt.answers)

            return TestResult(
                attempt_id=completed_attempt.id,
                score=completed_attempt.score,
                total_points=completed_attempt.total_points,
                earned_points=completed_attempt.earned_points,
                is_passed=is_passed,
                time_spent=completed_attempt.time_spent or 0,
                correct_answers=correct_answers,
                total_questions=total_questions
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ошибка при завершении теста: {str(e)}"
            )

    def get_test_statistics(self, test_id: int, current_user: User) -> TestStatistics:
        """Получить статистику по тесту"""
        test = self.test_crud.get_test(test_id)
        if not test:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Тест не найден"
            )

        # Только преподаватели и админы могут просматривать статистику
        if current_user.role not in ["teacher", "admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Недостаточно прав для просмотра статистики"
            )

        stats = self.test_crud.get_test_statistics(test_id)
        return TestStatistics(
            test_id=test_id,
            total_attempts=stats["total_attempts"],
            average_score=stats["average_score"],
            pass_rate=stats["pass_rate"],
            average_time=stats["average_time"]
        )

    def get_user_test_progress(self, test_id: int, current_user: User) -> UserTestProgress:
        """Получить прогресс пользователя по тесту"""
        test = self.test_crud.get_test(test_id)
        if not test:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Тест не найден"
            )

        progress = self.test_crud.get_user_test_progress(current_user.id, test_id)
        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Прогресс не найден"
            )

        return UserTestProgress(**progress)

    def get_user_attempts(self, test_id: int, current_user: User) -> List[TestAttempt]:
        """Получить все попытки пользователя для теста"""
        test = self.test_crud.get_test(test_id)
        if not test:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Тест не найден"
            )

        return self.test_crud.get_user_attempts(current_user.id, test_id)

    def _validate_test_data(self, test_data: TestCreate) -> None:
        """Валидация данных теста"""
        if not test_data.title.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Название теста не может быть пустым"
            )

        if test_data.time_limit and test_data.time_limit <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Время на прохождение должно быть положительным"
            )

        if test_data.max_attempts <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Максимальное количество попыток должно быть положительным"
            )

        if not (0 <= test_data.passing_score <= 100):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Проходной балл должен быть от 0 до 100"
            )

        self._validate_questions(test_data.questions)

    def _validate_questions(self, questions) -> None:
        """Валидация вопросов теста"""
        if not questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Тест должен содержать хотя бы один вопрос"
            )

        for i, question in enumerate(questions):
            if not question.question_text.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Текст вопроса {i+1} не может быть пустым"
                )

            if question.points <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Количество баллов за вопрос {i+1} должно быть положительным"
                )

            # Валидация вариантов ответов для multiple_choice и true_false
            if question.question_type in ["multiple_choice", "true_false"]:
                if not question.options:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Вопрос {i+1} должен содержать варианты ответов"
                    )

                correct_options = [opt for opt in question.options if opt.is_correct]
                if not correct_options:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Вопрос {i+1} должен содержать хотя бы один правильный ответ"
                    )

                if question.question_type == "true_false" and len(question.options) != 2:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Вопрос типа 'Да/Нет' {i+1} должен содержать ровно 2 варианта ответа"
                    )

            # Валидация для gesture_recognition
            if question.question_type == "gesture_recognition":
                if not question.gesture_class:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Для вопроса с распознаванием жестов {i+1} необходимо указать класс жеста"
                    )

    def _can_access_test(self, test, user: User) -> bool:
        """Проверка прав доступа к тесту"""
        # Преподаватели и админы имеют доступ ко всем тестам
        if user.role in ["teacher", "admin"]:
            return True

        # Студенты имеют доступ только к активным тестам
        return test.is_active


def get_test_service(db: Session) -> TestService:
    """Фабричная функция для создания экземпляра TestService"""
    return TestService(db)