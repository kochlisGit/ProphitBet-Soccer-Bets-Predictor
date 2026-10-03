"""Require fixtures to be confirmed by a current schedule refresh."""
from alembic import op
import sqlalchemy as sa

revision = "20260918_0003"
down_revision = "20260915_0002"
branch_labels = None
depends_on = None


def upgrade():
    # Preserve legacy rows for audit; the next successful scrape republishes
    # only fixtures observed in its provider response.
    op.add_column("fixtures", sa.Column("is_current", sa.Boolean(), server_default=sa.false(), nullable=False))


def downgrade():
    op.drop_column("fixtures", "is_current")
