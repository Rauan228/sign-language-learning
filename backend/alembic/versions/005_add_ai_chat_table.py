"""Add AI chat table

Revision ID: 005
Revises: 004
Create Date: 2024-01-15 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade():
    # Создание enum типов
    message_type_enum = postgresql.ENUM(
        'TEXT', 'GESTURE', 'IMAGE', 'VIDEO', 'AUDIO',
        name='messagetypeenum'
    )
    message_type_enum.create(op.get_bind())
    
    response_type_enum = postgresql.ENUM(
        'TEXT', 'IMAGE', 'VIDEO', 'GESTURE_DEMO', 'INTERACTIVE',
        name='responsetypeenum'
    )
    response_type_enum.create(op.get_bind())
    
    # Создание таблицы ai_chats
    op.create_table(
        'ai_chats',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.String(length=255), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('question_type', message_type_enum, nullable=False),
        sa.Column('question_file_url', sa.String(length=500), nullable=True),
        sa.Column('answer', sa.Text(), nullable=True),
        sa.Column('answer_type', response_type_enum, nullable=True),
        sa.Column('answer_file_url', sa.String(length=500), nullable=True),
        sa.Column('context_data', sa.JSON(), nullable=True),
        sa.Column('confidence_score', sa.Integer(), nullable=True),
        sa.Column('processing_time', sa.Integer(), nullable=True),
        sa.Column('user_rating', sa.Integer(), nullable=True),
        sa.Column('is_helpful', sa.String(length=10), nullable=True),
        sa.Column('user_feedback', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Создание индексов
    op.create_index('ix_ai_chats_user_id', 'ai_chats', ['user_id'])
    op.create_index('ix_ai_chats_session_id', 'ai_chats', ['session_id'])
    op.create_index('ix_ai_chats_created_at', 'ai_chats', ['created_at'])
    op.create_index('ix_ai_chats_user_session', 'ai_chats', ['user_id', 'session_id'])
    op.create_index('ix_ai_chats_question_type', 'ai_chats', ['question_type'])
    op.create_index('ix_ai_chats_user_rating', 'ai_chats', ['user_rating'])
    
    # Создание внешнего ключа
    op.create_foreign_key(
        'fk_ai_chats_user_id',
        'ai_chats', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # Добавление ограничений
    op.create_check_constraint(
        'ck_ai_chats_confidence_score',
        'ai_chats',
        'confidence_score >= 0 AND confidence_score <= 100'
    )
    
    op.create_check_constraint(
        'ck_ai_chats_user_rating',
        'ai_chats',
        'user_rating >= 1 AND user_rating <= 5'
    )
    
    op.create_check_constraint(
        'ck_ai_chats_processing_time',
        'ai_chats',
        'processing_time >= 0'
    )
    
    op.create_check_constraint(
        'ck_ai_chats_is_helpful',
        'ai_chats',
        "is_helpful IN ('yes', 'no')"
    )


def downgrade():
    # Удаление таблицы
    op.drop_table('ai_chats')
    
    # Удаление enum типов
    op.execute('DROP TYPE IF EXISTS messagetypeenum')
    op.execute('DROP TYPE IF EXISTS responsetypeenum')