"""Add factual data provenance fields.

Revision ID: 20260915_0002
Revises: 20260915_0001
"""

from alembic import op
import sqlalchemy as sa

revision = "20260915_0002"
down_revision = "20260915_0001"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("predictions", sa.Column("result_source", sa.Text(), nullable=True))
    op.add_column("predictions", sa.Column("result_verified_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("fixtures", sa.Column("source_url", sa.Text(), nullable=True))
    op.add_column("fixtures", sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True))
    # Legacy SA lines were generated from model probabilities and branded as
    # bookmaker observations. They cannot be made factual retroactively.
    op.execute("DELETE FROM sa_bookmaker_odds")


def downgrade():
    op.drop_column("fixtures", "fetched_at")
    op.drop_column("fixtures", "source_url")
    op.drop_column("predictions", "result_verified_at")
    op.drop_column("predictions", "result_source")
