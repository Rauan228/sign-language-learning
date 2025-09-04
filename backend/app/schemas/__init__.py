from .auth import (
    Token, TokenData, RefreshToken
)
from .user import (
    UserCreate, UserUpdate, UserResponse, UserLogin,
    UserProfile
)

from .course import (
    Course,
    CourseCreate,
    CourseUpdate,
    CourseInDB,
    CourseWithTeacher,
    CourseListResponse,
    CourseFilter,
    CourseStats,
    CourseLevel,
    SignLanguage,
    CourseStatus
)

from .lesson import (
    Lesson,
    LessonCreate,
    LessonUpdate,
    LessonInDB,
    LessonWithCourse,
    LessonListResponse,
    LessonFilter,
    LessonType,
    QuizQuestion,
    QuizSubmission,
    QuizResult,
    LessonProgress
)

from .progress import (
    CourseProgress,
    CourseProgressCreate,
    CourseProgressUpdate,
    CourseProgressInDB,
    CourseProgressWithDetails,
    LessonProgress as ProgressLessonProgress,
    LessonProgressCreate,
    LessonProgressUpdate,
    LessonProgressInDB,
    UserProgressSummary,
    CourseProgressStats,
    ProgressStatus,
    Certificate,
    CertificateCreate,
    ActivityLog
)

from .ai_chat import (
    ChatSession,
    ChatSessionCreate,
    ChatSessionUpdate,
    ChatSessionInDB,
    ChatSessionWithMessages,
    Message,
    MessageCreate,
    MessageUpdate,
    MessageInDB,
    ChatRequest,
    ChatResponse,
    GestureRecognitionRequest,
    GestureRecognitionResponse,
    MessageRole,
    MessageType,
    ChatSessionStatus,
    Feedback,
    FeedbackCreate
)

from .review import (
    Review,
    ReviewCreate,
    ReviewUpdate,
    ReviewInDB,
    ReviewWithAuthor,
    ReviewWithReplies,
    ReviewReply,
    ReviewReplyCreate,
    ReviewReplyUpdate,
    ReviewHelpfulness,
    ReviewHelpfulnessCreate,
    ReviewFilter,
    ReviewListResponse,
    ReviewStats,
    CourseReviewSummary,
    TeacherReviewSummary,
    ReviewType,
    ReviewStatus,
    ReviewReport,
    ReviewReportCreate
)

__all__ = [
    # User schemas
    "User",
    "UserCreate", 
    "UserUpdate",
    "UserInDB",
    "UserLogin",
    "UserRegister",
    "Token",
    "TokenData",
    "PasswordReset",
    "PasswordResetRequest",
    
    # Course schemas
    "Course",
    "CourseCreate",
    "CourseUpdate",
    "CourseInDB",
    "CourseWithTeacher",
    "CourseListResponse",
    "CourseFilter",
    "CourseStats",
    "CourseLevel",
    "SignLanguage",
    "CourseStatus",
    
    # Lesson schemas
    "Lesson",
    "LessonCreate",
    "LessonUpdate",
    "LessonInDB",
    "LessonWithCourse",
    "LessonListResponse",
    "LessonFilter",
    "LessonType",
    "QuizQuestion",
    "QuizSubmission",
    "QuizResult",
    "LessonProgress",
    
    # Progress schemas
    "CourseProgress",
    "CourseProgressCreate",
    "CourseProgressUpdate",
    "CourseProgressInDB",
    "CourseProgressWithDetails",
    "ProgressLessonProgress",
    "LessonProgressCreate",
    "LessonProgressUpdate",
    "LessonProgressInDB",
    "UserProgressSummary",
    "CourseProgressStats",
    "ProgressStatus",
    "Certificate",
    "CertificateCreate",
    "ActivityLog",
    
    # AI Chat schemas
    "ChatSession",
    "ChatSessionCreate",
    "ChatSessionUpdate",
    "ChatSessionInDB",
    "ChatSessionWithMessages",
    "Message",
    "MessageCreate",
    "MessageUpdate",
    "MessageInDB",
    "ChatRequest",
    "ChatResponse",
    "GestureRecognitionRequest",
    "GestureRecognitionResponse",
    "MessageRole",
    "MessageType",
    "ChatSessionStatus",
    "Feedback",
    "FeedbackCreate",
    
    # Review schemas
    "Review",
    "ReviewCreate",
    "ReviewUpdate",
    "ReviewInDB",
    "ReviewWithAuthor",
    "ReviewWithReplies",
    "ReviewReply",
    "ReviewReplyCreate",
    "ReviewReplyUpdate",
    "ReviewHelpfulness",
    "ReviewHelpfulnessCreate",
    "ReviewFilter",
    "ReviewListResponse",
    "ReviewStats",
    "CourseReviewSummary",
    "TeacherReviewSummary",
    "ReviewType",
    "ReviewStatus",
    "ReviewReport",
    "ReviewReportCreate"
]