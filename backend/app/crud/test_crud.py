from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc
from datetime import datetime

from app.models.test import Test, Question, QuestionOption, TestAttempt, Answer
from app.schemas.test import (
    TestCreate, TestUpdate, QuestionCreate, QuestionOptionCreate,
    TestAttemptCreate, AnswerCreate, TestAttemptSubmit
)


class TestCRUD:
    def __init__(self, db: Session):
        self.db = db

    # Test CRUD operations
    def get_test(self, test_id: int) -> Optional[Test]:
        """Получить тест по ID"""
        return self.db.query(Test).filter(Test.id == test_id).first()

    def get_tests_by_lesson(self, lesson_id: int, active_only: bool = True) -> List[Test]:
        """Получить все тесты урока"""
        query = self.db.query(Test).filter(Test.lesson_id == lesson_id)
        if active_only:
            query = query.filter(Test.is_active == True)
        return query.order_by(Test.created_at).all()

    def create_test(self, test_data: TestCreate) -> Test:
        """Создать новый тест"""
        # Создаем тест
        test = Test(
            title=test_data.title,
            description=test_data.description,
            lesson_id=test_data.lesson_id,
            time_limit=test_data.time_limit,
            max_attempts=test_data.max_attempts,
            passing_score=test_data.passing_score,
            is_active=test_data.is_active
        )
        self.db.add(test)
        self.db.flush()  # Получаем ID теста

        # Создаем вопросы
        for question_data in test_data.questions:
            question = Question(
                test_id=test.id,
                question_text=question_data.question_text,
                question_type=question_data.question_type,
                points=question_data.points,
                order_index=question_data.order_index,
                gesture_class=question_data.gesture_class
            )
            self.db.add(question)
            self.db.flush()  # Получаем ID вопроса

            # Создаем варианты ответов
            for option_data in question_data.options:
                option = QuestionOption(
                    question_id=question.id,
                    option_text=option_data.option_text,
                    is_correct=option_data.is_correct,
                    order_index=option_data.order_index
                )
                self.db.add(option)

        self.db.commit()
        self.db.refresh(test)
        return test

    def update_test(self, test_id: int, test_data: TestUpdate) -> Optional[Test]:
        """Обновить тест"""
        test = self.get_test(test_id)
        if not test:
            return None

        # Обновляем основные поля теста
        update_data = test_data.dict(exclude_unset=True, exclude={'questions'})
        for field, value in update_data.items():
            setattr(test, field, value)

        # Если переданы вопросы, обновляем их
        if test_data.questions is not None:
            # Удаляем старые вопросы
            self.db.query(Question).filter(Question.test_id == test_id).delete()
            
            # Создаем новые вопросы
            for question_data in test_data.questions:
                question = Question(
                    test_id=test.id,
                    question_text=question_data.question_text,
                    question_type=question_data.question_type,
                    points=question_data.points,
                    order_index=question_data.order_index,
                    gesture_class=question_data.gesture_class
                )
                self.db.add(question)
                self.db.flush()

                # Создаем варианты ответов
                for option_data in question_data.options:
                    option = QuestionOption(
                        question_id=question.id,
                        option_text=option_data.option_text,
                        is_correct=option_data.is_correct,
                        order_index=option_data.order_index
                    )
                    self.db.add(option)

        self.db.commit()
        self.db.refresh(test)
        return test

    def delete_test(self, test_id: int) -> bool:
        """Удалить тест"""
        test = self.get_test(test_id)
        if not test:
            return False
        
        self.db.delete(test)
        self.db.commit()
        return True

    # Test Attempt CRUD operations
    def create_attempt(self, user_id: int, test_id: int) -> Optional[TestAttempt]:
        """Создать новую попытку прохождения теста"""
        test = self.get_test(test_id)
        if not test:
            return None

        # Проверяем количество попыток пользователя
        attempts_count = self.db.query(TestAttempt).filter(
            and_(TestAttempt.test_id == test_id, TestAttempt.user_id == user_id)
        ).count()

        if attempts_count >= test.max_attempts:
            return None  # Превышено максимальное количество попыток

        attempt = TestAttempt(
            test_id=test_id,
            user_id=user_id
        )
        self.db.add(attempt)
        self.db.commit()
        self.db.refresh(attempt)
        return attempt

    def get_attempt(self, attempt_id: int) -> Optional[TestAttempt]:
        """Получить попытку по ID"""
        return self.db.query(TestAttempt).filter(TestAttempt.id == attempt_id).first()

    def get_user_attempts(self, user_id: int, test_id: int) -> List[TestAttempt]:
        """Получить все попытки пользователя для теста"""
        return self.db.query(TestAttempt).filter(
            and_(TestAttempt.test_id == test_id, TestAttempt.user_id == user_id)
        ).order_by(desc(TestAttempt.started_at)).all()

    def submit_attempt(self, attempt_id: int, submission: TestAttemptSubmit) -> Optional[TestAttempt]:
        """Завершить попытку прохождения теста"""
        attempt = self.get_attempt(attempt_id)
        if not attempt or attempt.is_completed:
            return None

        test = self.get_test(attempt.test_id)
        if not test:
            return None

        total_points = 0
        earned_points = 0
        correct_answers = 0

        # Обрабатываем ответы
        for answer_data in submission.answers:
            question = self.db.query(Question).filter(Question.id == answer_data.question_id).first()
            if not question:
                continue

            total_points += question.points
            is_correct = False
            points_earned = 0

            # Проверяем правильность ответа в зависимости от типа вопроса
            if question.question_type == "multiple_choice" and answer_data.selected_option_id:
                option = self.db.query(QuestionOption).filter(
                    QuestionOption.id == answer_data.selected_option_id
                ).first()
                if option and option.is_correct:
                    is_correct = True
                    points_earned = question.points
            elif question.question_type == "true_false" and answer_data.selected_option_id:
                option = self.db.query(QuestionOption).filter(
                    QuestionOption.id == answer_data.selected_option_id
                ).first()
                if option and option.is_correct:
                    is_correct = True
                    points_earned = question.points
            elif question.question_type == "text_input" and answer_data.text_answer:
                # Для текстовых ответов нужна дополнительная логика проверки
                # Пока считаем правильными все непустые ответы
                if answer_data.text_answer.strip():
                    is_correct = True
                    points_earned = question.points
            elif question.question_type == "gesture_recognition" and answer_data.gesture_result:
                # Проверяем соответствие распознанного жеста ожидаемому
                if answer_data.gesture_result == question.gesture_class:
                    is_correct = True
                    points_earned = question.points

            if is_correct:
                correct_answers += 1
                earned_points += points_earned

            # Сохраняем ответ
            answer = Answer(
                attempt_id=attempt_id,
                question_id=answer_data.question_id,
                selected_option_id=answer_data.selected_option_id,
                text_answer=answer_data.text_answer,
                gesture_result=answer_data.gesture_result,
                is_correct=is_correct,
                points_earned=points_earned
            )
            self.db.add(answer)

        # Вычисляем итоговый балл
        score = int((earned_points / total_points) * 100) if total_points > 0 else 0

        # Обновляем попытку
        attempt.score = score
        attempt.total_points = total_points
        attempt.earned_points = earned_points
        attempt.is_completed = True
        attempt.completed_at = datetime.utcnow()
        
        # Вычисляем время прохождения
        if attempt.started_at:
            time_spent = (datetime.utcnow() - attempt.started_at).total_seconds()
            attempt.time_spent = int(time_spent)

        self.db.commit()
        self.db.refresh(attempt)
        return attempt

    def get_test_statistics(self, test_id: int) -> dict:
        """Получить статистику по тесту"""
        attempts = self.db.query(TestAttempt).filter(
            and_(TestAttempt.test_id == test_id, TestAttempt.is_completed == True)
        ).all()

        if not attempts:
            return {
                "total_attempts": 0,
                "average_score": 0,
                "pass_rate": 0,
                "average_time": 0
            }

        test = self.get_test(test_id)
        passing_score = test.passing_score if test else 70

        total_attempts = len(attempts)
        passed_attempts = len([a for a in attempts if a.score >= passing_score])
        average_score = sum(a.score for a in attempts) / total_attempts
        pass_rate = (passed_attempts / total_attempts) * 100
        
        # Среднее время в минутах
        times = [a.time_spent for a in attempts if a.time_spent]
        average_time = (sum(times) / len(times) / 60) if times else 0

        return {
            "total_attempts": total_attempts,
            "average_score": round(average_score, 2),
            "pass_rate": round(pass_rate, 2),
            "average_time": round(average_time, 2)
        }

    def get_user_test_progress(self, user_id: int, test_id: int) -> dict:
        """Получить прогресс пользователя по тесту"""
        test = self.get_test(test_id)
        if not test:
            return None

        attempts = self.get_user_attempts(user_id, test_id)
        completed_attempts = [a for a in attempts if a.is_completed]
        
        best_score = max([a.score for a in completed_attempts]) if completed_attempts else None
        is_completed = any(a.score >= test.passing_score for a in completed_attempts) if completed_attempts else False
        last_attempt_date = attempts[0].started_at if attempts else None

        return {
            "test_id": test_id,
            "test_title": test.title,
            "attempts_count": len(attempts),
            "max_attempts": test.max_attempts,
            "best_score": best_score,
            "is_completed": is_completed,
            "last_attempt_date": last_attempt_date
        }