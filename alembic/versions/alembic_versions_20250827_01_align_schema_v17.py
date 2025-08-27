"""Align schema with ORM and code usage (features/backtest + core OMS tables).

Revision ID: 20250827_01
Revises: 20250825_07
Create Date: 2025-08-27
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20250827_01"
down_revision = "20250825_07"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)

    # Add missing feature columns used by code (adv_usd_21, overnight_gap, illiq_21, beta_63)
    fcols = [c["name"] for c in insp.get_columns("features")]
    add_map = {
        "adv_usd_21": sa.Float(),
        "overnight_gap": sa.Float(),
        "illiq_21": sa.Float(),
        "beta_63": sa.Float(),
    }
    for col, coltype in add_map.items():
        if col not in fcols:
            op.add_column("features", sa.Column(col, coltype, nullable=True))

    # Add tcost_impact to backtest_equity (default 0.0)
    bcols = [c["name"] for c in insp.get_columns("backtest_equity")]
    if "tcost_impact" not in bcols:
        op.add_column("backtest_equity", sa.Column("tcost_impact", sa.Float(), nullable=False, server_default="0.0"))

    # Create core tables if missing (as defined in db.py)
    tables = set(insp.get_table_names())

    if "shares_outstanding" not in tables:
        op.create_table(
            "shares_outstanding",
            sa.Column("symbol", sa.String(length=20), primary_key=True),
            sa.Column("as_of", sa.Date(), primary_key=True),
            sa.Column("shares", sa.BigInteger(), nullable=False),
            sa.Column("source", sa.String(length=32), nullable=True),
            sa.Column("knowledge_date", sa.Date(), nullable=True),
        )
        op.create_index("ix_shares_symbol_asof", "shares_outstanding", ["symbol", "as_of"])
        op.create_index("ix_shares_outstanding_bitemporal", "shares_outstanding", ["symbol", "as_of", "knowledge_date"])

    if "target_positions" not in tables:
        op.create_table(
            "target_positions",
            sa.Column("ts", sa.Date(), primary_key=True),
            sa.Column("symbol", sa.String(length=20), primary_key=True),
            sa.Column("weight", sa.Float(), nullable=False),
            sa.Column("price", sa.Float(), nullable=True),
            sa.Column("target_shares", sa.Integer(), nullable=True),
        )
        op.create_index("ix_target_positions_ts_symbol", "target_positions", ["ts", "symbol"])

    if "current_positions" not in tables:
        op.create_table(
            "current_positions",
            sa.Column("symbol", sa.String(length=20), primary_key=True),
            sa.Column("shares", sa.Integer(), nullable=False),
            sa.Column("market_value", sa.Float(), nullable=True),
            sa.Column("cost_basis", sa.Float(), nullable=True),
            sa.Column("last_updated", sa.DateTime(), nullable=True),
        )

    if "system_state" not in tables:
        op.create_table(
            "system_state",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("nav", sa.Float(), nullable=False),
            sa.Column("cash", sa.Float(), nullable=False),
            sa.Column("last_reconciled", sa.DateTime(), nullable=True),
        )

    if "task_status" not in tables:
        op.create_table(
            "task_status",
            sa.Column("task_id", sa.String(length=128), primary_key=True),
            sa.Column("task_name", sa.String(length=128), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("result", sa.JSON(), nullable=True),
            sa.Column("error_message", sa.String(length=1024), nullable=True),
            sa.Column("progress", sa.Integer(), nullable=True),
        )
        op.create_index("ix_task_status_created_at", "task_status", ["created_at"])

    if "data_validation_log" not in tables:
        op.create_table(
            "data_validation_log",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("run_timestamp", sa.DateTime(), nullable=False),
            sa.Column("validation_type", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("message", sa.String(length=1024), nullable=True),
            sa.Column("metrics", sa.JSON(), nullable=True),
            sa.Column("affected_symbols", sa.JSON(), nullable=True),
        )
        op.create_index("ix_validation_log_timestamp", "data_validation_log", ["run_timestamp"])
        op.create_index("ix_validation_log_type_status", "data_validation_log", ["validation_type", "status"])

    if "data_lineage" not in tables:
        op.create_table(
            "data_lineage",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("table_name", sa.String(length=64), nullable=False),
            sa.Column("symbol", sa.String(length=20), nullable=True),
            sa.Column("data_date", sa.Date(), nullable=False),
            sa.Column("ingestion_timestamp", sa.DateTime(), nullable=False),
            sa.Column("source", sa.String(length=32), nullable=False),
            sa.Column("source_timestamp", sa.DateTime(), nullable=True),
            sa.Column("quality_score", sa.Float(), nullable=True),
            sa.Column("lineage_metadata", sa.JSON(), nullable=True),
        )
        op.create_index("ix_lineage_table_symbol_date", "data_lineage", ["table_name", "symbol", "data_date"])
        op.create_index("ix_lineage_ingestion_time", "data_lineage", ["ingestion_timestamp"])


def downgrade() -> None:
    # Best-effort reversible changes
    try:
        op.drop_index("ix_lineage_ingestion_time", table_name="data_lineage")
        op.drop_index("ix_lineage_table_symbol_date", table_name="data_lineage")
        op.drop_table("data_lineage")
    except Exception:
        pass
    try:
        op.drop_index("ix_validation_log_type_status", table_name="data_validation_log")
        op.drop_index("ix_validation_log_timestamp", table_name="data_validation_log")
        op.drop_table("data_validation_log")
    except Exception:
        pass
    try:
        op.drop_index("ix_task_status_created_at", table_name="task_status")
        op.drop_table("task_status")
    except Exception:
        pass
    try:
        op.drop_table("system_state")
    except Exception:
        pass
    try:
        op.drop_table("current_positions")
    except Exception:
        pass
    try:
        op.drop_index("ix_target_positions_ts_symbol", table_name="target_positions")
        op.drop_table("target_positions")
    except Exception:
        pass
    try:
        op.drop_index("ix_shares_outstanding_bitemporal", table_name="shares_outstanding")
        op.drop_index("ix_shares_symbol_asof", table_name="shares_outstanding")
        op.drop_table("shares_outstanding")
    except Exception:
        pass
    try:
        op.drop_column("backtest_equity", "tcost_impact")
    except Exception:
        pass
    for col in ("beta_63", "illiq_21", "overnight_gap", "adv_usd_21"):
        try:
            op.drop_column("features", col)
        except Exception:
            pass