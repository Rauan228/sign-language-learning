from typing import List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    TEXT_INPUT = "text_input"
    GESTURE_RECOGNITION = "gesture_recognition"


# Question Option schemas
class QuestionOptionBase(BaseModel):
    option_text: str
    is_correct: bool = False
    order_index: int = 0


class QuestionOptionCreate(QuestionOptionBase):
    pass


class QuestionOptionUpdate(BaseModel):
    option_text: Optional[str] = None
    is_correct: Optional[bool] = None
    order_index: Optional[int] = None


class QuestionOption(QuestionOptionBase):
    id: int
    question_id: int

    class Config:
        from_attributes = True


# Question schemas
class QuestionBase(BaseModel):
    question_text: str
    question_type: QuestionType
    points: int = 1
    order_index: int = 0
    gesture_class: Optional[str] = None


class QuestionCreate(QuestionBase):
    options: List[QuestionOptionCreate] = []


class QuestionUpdate(BaseModel):
    question_text: Optional[str] = None
    question_type: Optional[QuestionType] = None
    points: Optional[int] = None
    order_index: Optional[int] = None
    gesture_class: Optional[str] = None
    options: Optional[List[QuestionOptionCreate]] = None


class Question(QuestionBase):
    id: int
    test_id: int
    created_at: datetime
    options: List[QuestionOption] = []

    class Config:
        from_attributes = True


# Test schemas
class TestBase(BaseModel):
    title: str
    description: Optional[str] = None
    time_limit: Optional[int] = None  # в минутах
    max_attempts: int = 3
    passing_score: int = 70  # процент правильных ответов
    is_active: bool = True


class TestCreate(TestBase):
    lesson_id: int
    questions: List[QuestionCreate] = []


class TestUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    time_limit: Optional[int] = None
    max_attempts: Optional[int] = None
    passing_score: Optional[int] = None
    is_active: Optional[bool] = None
    questions: Optional[List[QuestionCreate]] = None


class Test(TestBase):
    id: int
    lesson_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    questions: List[Question] = []

    class Config:
        from_attributes = True


# Answer schemas
class AnswerBase(BaseModel):
    question_id: int
    selected_option_id: Optional[int] = None
    text_answer: Optional[str] = None
    gesture_result: Optional[str] = None


class AnswerCreate(AnswerBase):
    pass


class Answer(AnswerBase):
    id: int
    attempt_id: int
    is_correct: Optional[bool] = None
    points_earned: int = 0
    answered_at: datetime

    class Config:
        from_attributes = True


# Test Attempt schemas
class TestAttemptBase(BaseModel):
    test_id: int


class TestAttemptCreate(TestAttemptBase):
    pass


class TestAttemptSubmit(BaseModel):
    answers: List[AnswerCreate]


class TestAttempt(TestAttemptBase):
    id: int
    user_id: int
    score: Optional[int] = None
    total_points: Optional[int] = None
    earned_points: Optional[int] = None
    is_completed: bool = False
    started_at: datetime
    completed_at: Optional[datetime] = None
    time_spent: Optional[int] = None  # в секундах
    answers: List[Answer] = []

    class Config:
        from_attributes = True


# Response schemas
class TestResult(BaseModel):
    attempt_id: int
    score: int
    total_points: int
    earned_points: int
    is_passed: bool
    time_spent: int
    correct_answers: int
    total_questions: int


class TestStatistics(BaseModel):
    test_id: int
    total_attempts: int
    average_score: float
    pass_rate: float  # процент успешных попыток
    average_time: float  # среднее время прохождения в минутах


class UserTestProgress(BaseModel):
    test_id: int
    test_title: str
    attempts_count: int
    max_attempts: int
    best_score: Optional[int] = None
    is_completed: bool = False
    last_attempt_date: Optional[datetime] = None