"""Add certificates table

Revision ID: 004
Revises: 003
Create Date: 2024-01-15 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Создание таблицы сертификатов
    op.create_table('certificates',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('course_id', sa.Integer(), nullable=False),
        sa.Column('certificate_number', sa.String(length=50), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('completion_date', sa.DateTime(), nullable=False),
        sa.Column('total_score', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('max_possible_score', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('percentage', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_time_spent', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Создание индексов
    op.create_index('ix_certificates_id', 'certificates', ['id'], unique=False)
    op.create_index('ix_certificates_user_id', 'certificates', ['user_id'], unique=False)
    op.create_index('ix_certificates_course_id', 'certificates', ['course_id'], unique=False)
    op.create_index('ix_certificates_certificate_number', 'certificates', ['certificate_number'], unique=True)
    op.create_index('ix_certificates_completion_date', 'certificates', ['completion_date'], unique=False)
    op.create_index('ix_certificates_is_active', 'certificates', ['is_active'], unique=False)
    
    # Создание составного индекса для проверки уникальности активных сертификатов
    op.create_index(
        'ix_certificates_user_course_active', 
        'certificates', 
        ['user_id', 'course_id', 'is_active'], 
        unique=False
    )
    
    # Добавление внешних ключей
    op.create_foreign_key(
        'fk_certificates_user_id', 
        'certificates', 
        'users', 
        ['user_id'], 
        ['id'], 
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_certificates_course_id', 
        'certificates', 
        'courses', 
        ['course_id'], 
        ['id'], 
        ondelete='CASCADE'
    )


def downgrade() -> None:
    # Удаление внешних ключей
    op.drop_constraint('fk_certificates_course_id', 'certificates', type_='foreignkey')
    op.drop_constraint('fk_certificates_user_id', 'certificates', type_='foreignkey')
    
    # Удаление индексов
    op.drop_index('ix_certificates_user_course_active', table_name='certificates')
    op.drop_index('ix_certificates_is_active', table_name='certificates')
    op.drop_index('ix_certificates_completion_date', table_name='certificates')
    op.drop_index('ix_certificates_certificate_number', table_name='certificates')
    op.drop_index('ix_certificates_course_id', table_name='certificates')
    op.drop_index('ix_certificates_user_id', table_name='certificates')
    op.drop_index('ix_certificates_id', table_name='certificates')
    
    # Удаление таблицы
    op.drop_table('certificates')