"""Preserve accumulator legs and distinguish observed from manual odds."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "20260924_0005"
down_revision = "20260919_0004"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("user_bet_slips", sa.Column("legs", JSONB(), nullable=True))
    op.add_column("user_bet_slips", sa.Column("bookmaker", sa.String(50), nullable=True))
    op.add_column("user_bet_slips", sa.Column("odds_origin", sa.String(30), nullable=False, server_default="manual_unverified"))


def downgrade():
    for column in ("odds_origin", "bookmaker", "legs"):
        op.drop_column("user_bet_slips", column)
