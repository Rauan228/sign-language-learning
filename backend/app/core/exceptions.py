"""Пользовательские исключения для приложения"""

class BaseAppException(Exception):
    """Базовое исключение приложения"""
    def __init__(self, message: str, code: str = None):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ValidationError(BaseAppException):
    """Ошибка валидации данных"""
    pass


class AuthenticationError(BaseAppException):
    """Ошибка аутентификации"""
    pass


class AuthorizationError(BaseAppException):
    """Ошибка авторизации"""
    pass


class NotFoundError(BaseAppException):
    """Ошибка - ресурс не найден"""
    pass


class AlreadyExistsError(BaseAppException):
    """Ошибка - ресурс уже существует"""
    pass


class PermissionDeniedError(BaseAppException):
    """Ошибка - доступ запрещен"""
    pass


# Исключения для пользователей
class UserNotFoundError(NotFoundError):
    """Пользователь не найден"""
    pass


class UserAlreadyExistsError(AlreadyExistsError):
    """Пользователь уже существует"""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Неверные учетные данные"""
    pass


# Исключения для курсов
class CourseNotFoundError(NotFoundError):
    """Курс не найден"""
    pass


class CourseAlreadyExistsError(AlreadyExistsError):
    """Курс уже существует"""
    pass


class CourseAccessDeniedError(PermissionDeniedError):
    """Доступ к курсу запрещен"""
    pass


# Исключения для уроков
class LessonNotFoundError(NotFoundError):
    """Урок не найден"""
    pass


class LessonAlreadyExistsError(AlreadyExistsError):
    """Урок уже существует"""
    pass


# Исключения для записи на курсы
class EnrollmentError(BaseAppException):
    """Ошибка записи на курс"""
    pass


class AlreadyEnrolledError(EnrollmentError):
    """Пользователь уже записан на курс"""
    pass


class NotEnrolledError(EnrollmentError):
    """Пользователь не записан на курс"""
    pass


# Исключения для прогресса
class ProgressNotFoundError(NotFoundError):
    """Прогресс не найден"""
    pass


# Исключения для отзывов
class ReviewNotFoundError(NotFoundError):
    """Отзыв не найден"""
    pass


class ReviewAlreadyExistsError(AlreadyExistsError):
    """Отзыв уже существует"""
    pass


# Исключения для AI чата
class AIChatError(BaseAppException):
    """Ошибка AI чата"""
    pass


class ChatSessionNotFoundError(NotFoundError):
    """Сессия чата не найдена"""
    pass


# Исключения для файлов
class FileUploadError(BaseAppException):
    """Ошибка загрузки файла"""
    pass


class FileNotFoundError(NotFoundError):
    """Файл не найден"""
    pass


class FileSizeExceededError(FileUploadError):
    """Превышен размер файла"""
    pass


class InvalidFileTypeError(FileUploadError):
    """Недопустимый тип файла"""
    pass


# Исключения для базы данных
class DatabaseError(BaseAppException):
    """Ошибка базы данных"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Ошибка подключения к базе данных"""
    pass


# Исключения для внешних сервисов
class ExternalServiceError(BaseAppException):
    """Ошибка внешнего сервиса"""
    pass


class EmailServiceError(ExternalServiceError):
    """Ошибка сервиса электронной почты"""
    pass


class PaymentServiceError(ExternalServiceError):
    """Ошибка платежного сервиса"""
    pass