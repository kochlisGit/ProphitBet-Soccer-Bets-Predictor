"""Reserve training capacity and withdraw unverified legacy accuracy metrics."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "20260919_0004"
down_revision = "20260918_0003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("training_jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()))
    op.create_index("ix_training_jobs_user_id", "training_jobs", ["user_id"])
    op.execute("""UPDATE trained_models SET metrics = jsonb_build_object(
        'evaluation', 'legacy_unverified', 'legacy_reported', metrics)
        WHERE metrics IS NOT NULL AND metrics->>'evaluation' IS DISTINCT FROM 'held_out_only'""")


def downgrade():
    op.drop_table("training_jobs")
