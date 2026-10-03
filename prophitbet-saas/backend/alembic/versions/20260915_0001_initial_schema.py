"""Initial ProphitBet SaaS schema.

Revision ID: 20260915_0001
Revises:
"""

from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260915_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    if not context.is_offline_mode():
        expected_tables = {
            "users", "subscriptions", "leagues", "league_datasets", "trained_models",
            "predictions", "mirofish_simulations", "fixtures", "api_usage",
            "user_bet_slips", "sa_bookmaker_odds",
        }
        existing_tables = set(sa.inspect(op.get_bind()).get_table_names())
        if expected_tables.issubset(existing_tables):
            # Adopt databases created by the pre-Alembic startup path. Alembic
            # records this revision after upgrade() returns.
            return
        if existing_tables.intersection(expected_tables):
            raise RuntimeError(
                "Partial legacy schema detected; repair it before applying the initial migration"
            )

    uuid = postgresql.UUID(as_uuid=True)
    jsonb = postgresql.JSONB(astext_type=sa.Text())

    op.create_table("users",
        sa.Column("id", uuid, nullable=False), sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.String(128)), sa.Column("name", sa.String(120), nullable=False),
        sa.Column("avatar_url", sa.Text()), sa.Column("provider", sa.String(20), nullable=False),
        sa.Column("stripe_customer_id", sa.String(64)), sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_admin", sa.Boolean(), nullable=False), sa.Column("preferences", jsonb),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"), sa.UniqueConstraint("stripe_customer_id"))
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table("leagues",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False), sa.Column("country", sa.String(100), nullable=False),
        sa.Column("name", sa.String(200), nullable=False), sa.Column("category", sa.String(20), nullable=False),
        sa.Column("url", sa.Text(), nullable=False), sa.Column("fixture_url", sa.Text()),
        sa.Column("start_year", sa.Integer(), nullable=False), sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("last_synced_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"))
    op.create_index("ix_leagues_country_name", "leagues", ["country", "name"], unique=True)

    op.create_table("subscriptions",
        sa.Column("id", uuid, nullable=False), sa.Column("user_id", uuid, nullable=False),
        sa.Column("plan", sa.String(20), nullable=False), sa.Column("stripe_subscription_id", sa.String(64)),
        sa.Column("status", sa.String(20), nullable=False), sa.Column("current_period_start", sa.DateTime(timezone=True)),
        sa.Column("current_period_end", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("stripe_subscription_id"), sa.UniqueConstraint("user_id"))

    op.create_table("league_datasets",
        sa.Column("id", uuid, nullable=False), sa.Column("league_id", sa.Integer(), nullable=False),
        sa.Column("season", sa.String(20)), sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["league_id"], ["leagues.id"], ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"))
    op.create_index("ix_league_datasets_league_created", "league_datasets", ["league_id", "created_at"])

    op.create_table("trained_models",
        sa.Column("id", uuid, nullable=False), sa.Column("user_id", uuid), sa.Column("league_id", sa.Integer(), nullable=False),
        sa.Column("model_type", sa.String(50), nullable=False), sa.Column("target_type", sa.String(30), nullable=False),
        sa.Column("hyperparams", jsonb), sa.Column("metrics", jsonb), sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("is_public", sa.Boolean(), nullable=False), sa.Column("is_house_model", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["league_id"], ["leagues.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"), sa.PrimaryKeyConstraint("id"))
    op.create_index("ix_trained_models_league_house", "trained_models", ["league_id", "is_house_model"])

    op.create_table("fixtures",
        sa.Column("id", uuid, nullable=False), sa.Column("league_id", sa.Integer(), nullable=False),
        sa.Column("home_team", sa.String(200), nullable=False), sa.Column("away_team", sa.String(200), nullable=False),
        sa.Column("match_date", sa.DateTime(timezone=True)), sa.Column("odds_1", sa.Float()),
        sa.Column("odds_x", sa.Float()), sa.Column("odds_2", sa.Float()), sa.Column("predicted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["league_id"], ["leagues.id"], ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"))
    op.create_index("ix_fixtures_league_date", "fixtures", ["league_id", "match_date"])
    op.create_index("ix_fixtures_league_predicted", "fixtures", ["league_id", "predicted"])

    op.create_table("api_usage",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False), sa.Column("user_id", uuid, nullable=False),
        sa.Column("endpoint", sa.String(200), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"))
    op.create_index("ix_api_usage_user_ts", "api_usage", ["user_id", "timestamp"])

    op.create_table("predictions",
        sa.Column("id", uuid, nullable=False), sa.Column("model_id", uuid, nullable=False),
        sa.Column("league_id", sa.Integer(), nullable=False), sa.Column("home_team", sa.String(200), nullable=False),
        sa.Column("away_team", sa.String(200), nullable=False), sa.Column("match_date", sa.DateTime(timezone=True)),
        sa.Column("predicted_result", sa.String(10), nullable=False), sa.Column("probabilities", jsonb),
        sa.Column("actual_result", sa.String(10)), sa.Column("actual_score", sa.String(20)), sa.Column("is_correct", sa.Boolean()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["league_id"], ["leagues.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_id"], ["trained_models.id"], ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"))
    op.create_index("ix_predictions_date", "predictions", ["match_date"])
    op.create_index("ix_predictions_league_created", "predictions", ["league_id", "created_at"])
    op.create_index("ix_predictions_league_match", "predictions", ["league_id", "match_date"])

    op.create_table("mirofish_simulations",
        sa.Column("id", uuid, nullable=False), sa.Column("prediction_id", uuid, nullable=False),
        sa.Column("swarm_predicted_result", sa.String(10), nullable=False), sa.Column("swarm_probabilities", jsonb, nullable=False),
        sa.Column("ensemble_predicted_result", sa.String(10), nullable=False), sa.Column("ensemble_probabilities", jsonb, nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False), sa.Column("consensus_level", sa.String(50), nullable=False),
        sa.Column("simulation_report", jsonb, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"), sa.UniqueConstraint("prediction_id"))
    op.create_index("ix_mirofish_consensus", "mirofish_simulations", ["consensus_level"])
    op.create_index("ix_mirofish_pred_id", "mirofish_simulations", ["prediction_id"])

    op.create_table("user_bet_slips",
        sa.Column("id", uuid, nullable=False), sa.Column("user_id", uuid, nullable=False), sa.Column("prediction_id", uuid),
        sa.Column("match_title", sa.String(250), nullable=False), sa.Column("league_name", sa.String(100), nullable=False),
        sa.Column("selection", sa.String(20), nullable=False), sa.Column("odds_taken", sa.Float(), nullable=False),
        sa.Column("stake_amount", sa.Float(), nullable=False), sa.Column("status", sa.String(20), nullable=False),
        sa.Column("pnl", sa.Float(), nullable=False), sa.Column("closing_odds", sa.Float()), sa.Column("clv_edge_pct", sa.Float()),
        sa.Column("notes", sa.Text()), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("settled_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"))
    op.create_index("ix_bet_slips_user_created", "user_bet_slips", ["user_id", "created_at"])
    op.create_index("ix_bet_slips_user_status", "user_bet_slips", ["user_id", "status"])

    op.create_table("sa_bookmaker_odds",
        sa.Column("id", uuid, nullable=False), sa.Column("fixture_id", uuid), sa.Column("prediction_id", uuid),
        sa.Column("match_title", sa.String(250), nullable=False), sa.Column("home_team", sa.String(120), nullable=False),
        sa.Column("away_team", sa.String(120), nullable=False), sa.Column("league_name", sa.String(100), nullable=False),
        sa.Column("match_date", sa.DateTime(timezone=True)), sa.Column("bookmaker", sa.String(50), nullable=False),
        sa.Column("market_type", sa.String(50), nullable=False), sa.Column("odds_home", sa.Float()),
        sa.Column("odds_draw", sa.Float()), sa.Column("odds_away", sa.Float()), sa.Column("over_15", sa.Float()),
        sa.Column("under_15", sa.Float()), sa.Column("over_25", sa.Float()), sa.Column("under_25", sa.Float()),
        sa.Column("over_35", sa.Float()), sa.Column("under_35", sa.Float()), sa.Column("btts_yes", sa.Float()),
        sa.Column("btts_no", sa.Float()), sa.Column("dc_1x", sa.Float()), sa.Column("dc_12", sa.Float()),
        sa.Column("dc_x2", sa.Float()), sa.Column("dnb_home", sa.Float()), sa.Column("dnb_away", sa.Float()),
        sa.Column("markets_data", jsonb), sa.Column("margin_pct", sa.Float()), sa.Column("source_url", sa.String(300)),
        sa.Column("scraped_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["fixture_id"], ["fixtures.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"], ondelete="SET NULL"), sa.PrimaryKeyConstraint("id"))
    op.create_index("ix_sa_odds_league_date", "sa_bookmaker_odds", ["league_name", "match_date"])
    op.create_index("ix_sa_odds_match_bookmaker", "sa_bookmaker_odds", ["match_title", "bookmaker"])
    op.create_index("ix_sa_odds_scraped_at", "sa_bookmaker_odds", ["scraped_at"])


def downgrade():
    for table in ("sa_bookmaker_odds", "user_bet_slips", "mirofish_simulations", "predictions",
                  "api_usage", "fixtures", "trained_models", "league_datasets", "subscriptions", "leagues", "users"):
        op.drop_table(table)
