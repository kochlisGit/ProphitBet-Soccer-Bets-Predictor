import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer,
    String, Text, func, select,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship, column_property

from backend.app.db.session import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    avatar_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    provider: Mapped[str] = mapped_column(String(20), nullable=False, default="email")
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    preferences: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    subscription: Mapped[Optional["Subscription"]] = relationship(back_populates="user", uselist=False, lazy="selectin")
    trained_models: Mapped[list["TrainedModel"]] = relationship(back_populates="user", lazy="selectin")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    plan: Mapped[str] = mapped_column(String(20), nullable=False, default="free")
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, unique=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    current_period_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    current_period_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="subscription")


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(20), nullable=False, default="main")
    url: Mapped[str] = mapped_column(Text, nullable=False)
    fixture_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    start_year: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_leagues_country_name", "country", "name", unique=True),)

    datasets: Mapped[list["LeagueDataset"]] = relationship(back_populates="league", lazy="selectin")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="league")


class LeagueDataset(Base):
    __tablename__ = "league_datasets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False)
    season: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    league: Mapped["League"] = relationship(back_populates="datasets")

    __table_args__ = (
        Index("ix_league_datasets_league_created", "league_id", "created_at"),
    )


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    target_type: Mapped[str] = mapped_column(String(30), nullable=False, default="result")
    hyperparams: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    is_house_model: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[Optional["User"]] = relationship(back_populates="trained_models")
    league: Mapped["League"] = relationship()

    __table_args__ = (
        Index("ix_trained_models_league_house", "league_id", "is_house_model"),
    )


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("trained_models.id", ondelete="CASCADE"), nullable=False)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False)
    home_team: Mapped[str] = mapped_column(String(200), nullable=False)
    away_team: Mapped[str] = mapped_column(String(200), nullable=False)
    match_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    predicted_result: Mapped[str] = mapped_column(String(10), nullable=False)
    probabilities: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    actual_result: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    actual_score: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    result_source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_predictions_date", "match_date"),
        Index("ix_predictions_league_created", "league_id", "created_at"),
        Index("ix_predictions_league_match", "league_id", "match_date"),
    )

    # Derived from the model's immutable market identity; no duplicate schema state.
    market_type = column_property(
        select(TrainedModel.target_type).where(TrainedModel.id == model_id)
        .correlate_except(TrainedModel).scalar_subquery()
    )
    model: Mapped["TrainedModel"] = relationship()
    league: Mapped["League"] = relationship(back_populates="predictions")
    mirofish_simulation: Mapped[Optional["MiroFishSimulation"]] = relationship(
        back_populates="prediction", uselist=False, cascade="all, delete-orphan", lazy="selectin"
    )


class MiroFishSimulation(Base):
    __tablename__ = "mirofish_simulations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    swarm_predicted_result: Mapped[str] = mapped_column(String(10), nullable=False)
    swarm_probabilities: Mapped[dict] = mapped_column(JSONB, nullable=False)
    ensemble_predicted_result: Mapped[str] = mapped_column(String(10), nullable=False)
    ensemble_probabilities: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    consensus_level: Mapped[str] = mapped_column(String(50), nullable=False, default="MODERATE_AGREEMENT")
    simulation_report: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    prediction: Mapped["Prediction"] = relationship(back_populates="mirofish_simulation")

    __table_args__ = (
        Index("ix_mirofish_pred_id", "prediction_id"),
        Index("ix_mirofish_consensus", "consensus_level"),
    )


class Fixture(Base):
    __tablename__ = "fixtures"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False)
    home_team: Mapped[str] = mapped_column(String(200), nullable=False)
    away_team: Mapped[str] = mapped_column(String(200), nullable=False)
    match_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    odds_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    fetched_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    predicted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_fixtures_league_date", "league_id", "match_date"),
        Index("ix_fixtures_league_predicted", "league_id", "predicted"),
    )

    league: Mapped["League"] = relationship()


class ApiUsage(Base):
    __tablename__ = "api_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(200), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_api_usage_user_ts", "user_id", "timestamp"),)


class UserBetSlip(Base):
    __tablename__ = "user_bet_slips"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    prediction_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True)
    match_title: Mapped[str] = mapped_column(String(250), nullable=False)
    league_name: Mapped[str] = mapped_column(String(100), nullable=False, default="League")
    selection: Mapped[str] = mapped_column(String(20), nullable=False)
    legs: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    bookmaker: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    odds_origin: Mapped[str] = mapped_column(String(30), nullable=False, default="manual_unverified", server_default="manual_unverified")
    odds_taken: Mapped[float] = mapped_column(Float, nullable=False)
    stake_amount: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    closing_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clv_edge_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    settled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_bet_slips_user_created", "user_id", "created_at"),
        Index("ix_bet_slips_user_status", "user_id", "status"),
    )

    user: Mapped["User"] = relationship()
    prediction: Mapped[Optional["Prediction"]] = relationship()


class SABookmakerOdds(Base):
    __tablename__ = "sa_bookmaker_odds"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fixture_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("fixtures.id", ondelete="SET NULL"), nullable=True)
    prediction_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True)
    match_title: Mapped[str] = mapped_column(String(250), nullable=False)
    home_team: Mapped[str] = mapped_column(String(120), nullable=False)
    away_team: Mapped[str] = mapped_column(String(120), nullable=False)
    league_name: Mapped[str] = mapped_column(String(100), nullable=False, default="Betway Premiership")
    match_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    bookmaker: Mapped[str] = mapped_column(String(50), nullable=False)  # HOLLYWOODBETS, BETWAY
    market_type: Mapped[str] = mapped_column(String(50), nullable=False, default="1X2")

    # 1X2 Market Odds
    odds_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_draw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_away: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Over / Under Goals
    over_15: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    under_15: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    over_25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    under_25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    over_35: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    under_35: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Both Teams to Score (BTTS)
    btts_yes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    btts_no: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Double Chance
    dc_1x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dc_12: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dc_x2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Draw No Bet (DNB)
    dnb_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dnb_away: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Full Exotic Markets Data (Correct Scores, Asian Handicaps, HT/FT, Clean Sheet)
    markets_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    margin_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    scraped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_sa_odds_match_bookmaker", "match_title", "bookmaker"),
        Index("ix_sa_odds_league_date", "league_name", "match_date"),
        Index("ix_sa_odds_scraped_at", "scraped_at"),
    )


