"""Initial database schema

Revision ID: 17500dfe2f3a
Revises: 
Create Date: 2025-12-12 14:57:22.371778

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '17500dfe2f3a'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('role', sa.Enum('ADMIN', 'USER', 'VIEWER', name='userrole'), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('organization', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('api_key', sa.String(length=64), nullable=True),
        sa.Column('api_key_created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_api_key'), 'users', ['api_key'], unique=True)

    # Create spark_applications table
    op.create_table(
        'spark_applications',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('app_id', sa.String(), nullable=False),
        sa.Column('app_name', sa.String(), nullable=False),
        sa.Column('user', sa.String(), nullable=True),
        sa.Column('submit_time', sa.DateTime(), nullable=True),
        sa.Column('start_time', sa.DateTime(), nullable=True),
        sa.Column('end_time', sa.DateTime(), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('spark_version', sa.String(), nullable=True),
        sa.Column('executor_cores', sa.Integer(), nullable=True),
        sa.Column('executor_memory_mb', sa.Integer(), nullable=True),
        sa.Column('num_executors', sa.Integer(), nullable=True),
        sa.Column('driver_memory_mb', sa.Integer(), nullable=True),
        sa.Column('total_tasks', sa.Integer(), nullable=True),
        sa.Column('failed_tasks', sa.Integer(), nullable=True),
        sa.Column('total_stages', sa.Integer(), nullable=True),
        sa.Column('failed_stages', sa.Integer(), nullable=True),
        sa.Column('input_bytes', sa.Integer(), nullable=True),
        sa.Column('output_bytes', sa.Integer(), nullable=True),
        sa.Column('shuffle_read_bytes', sa.Integer(), nullable=True),
        sa.Column('shuffle_write_bytes', sa.Integer(), nullable=True),
        sa.Column('cpu_time_ms', sa.Integer(), nullable=True),
        sa.Column('memory_spilled_bytes', sa.Integer(), nullable=True),
        sa.Column('disk_spilled_bytes', sa.Integer(), nullable=True),
        sa.Column('executor_run_time_ms', sa.Integer(), nullable=True),
        sa.Column('executor_cpu_time_ms', sa.Integer(), nullable=True),
        sa.Column('jvm_gc_time_ms', sa.Integer(), nullable=True),
        sa.Column('peak_memory_usage', sa.Integer(), nullable=True),
        sa.Column('cluster_type', sa.String(), nullable=True),
        sa.Column('instance_type', sa.String(), nullable=True),
        sa.Column('estimated_cost', sa.Float(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('environment', sa.JSON(), nullable=True),
        sa.Column('spark_configs', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_spark_applications_id'), 'spark_applications', ['id'], unique=False)
    op.create_index(op.f('ix_spark_applications_app_id'), 'spark_applications', ['app_id'], unique=True)

    # Create spark_stages table
    op.create_table(
        'spark_stages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('application_id', sa.Integer(), nullable=False),
        sa.Column('stage_id', sa.Integer(), nullable=False),
        sa.Column('stage_name', sa.String(), nullable=True),
        sa.Column('num_tasks', sa.Integer(), nullable=True),
        sa.Column('submission_time', sa.DateTime(), nullable=True),
        sa.Column('completion_time', sa.DateTime(), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('executor_run_time_ms', sa.Integer(), nullable=True),
        sa.Column('executor_cpu_time_ms', sa.Integer(), nullable=True),
        sa.Column('memory_bytes_spilled', sa.Integer(), nullable=True),
        sa.Column('disk_bytes_spilled', sa.Integer(), nullable=True),
        sa.Column('input_bytes', sa.Integer(), nullable=True),
        sa.Column('output_bytes', sa.Integer(), nullable=True),
        sa.Column('shuffle_read_bytes', sa.Integer(), nullable=True),
        sa.Column('shuffle_write_bytes', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['application_id'], ['spark_applications.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create job_recommendations table
    op.create_table(
        'job_recommendations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_signature', sa.String(), nullable=True),
        sa.Column('recommended_executor_cores', sa.Integer(), nullable=True),
        sa.Column('recommended_executor_memory_mb', sa.Integer(), nullable=True),
        sa.Column('recommended_num_executors', sa.Integer(), nullable=True),
        sa.Column('recommended_driver_memory_mb', sa.Integer(), nullable=True),
        sa.Column('predicted_duration_ms', sa.Integer(), nullable=True),
        sa.Column('predicted_cost', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('recommendation_method', sa.String(), nullable=True),
        sa.Column('similar_jobs', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('used_at', sa.DateTime(), nullable=True),
        sa.Column('feedback_score', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_job_recommendations_job_signature'), 'job_recommendations', ['job_signature'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_job_recommendations_job_signature'), table_name='job_recommendations')
    op.drop_table('job_recommendations')
    op.drop_table('spark_stages')
    op.drop_index(op.f('ix_spark_applications_app_id'), table_name='spark_applications')
    op.drop_index(op.f('ix_spark_applications_id'), table_name='spark_applications')
    op.drop_table('spark_applications')
    op.drop_index(op.f('ix_users_api_key'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_table('users')
    sa.Enum('ADMIN', 'USER', 'VIEWER', name='userrole').drop(op.get_bind())
