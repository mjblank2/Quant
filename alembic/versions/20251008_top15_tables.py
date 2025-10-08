"""top15 holdings + intents tables"""

from alembic import op
import sqlalchemy as sa

# Generate a unique revision id, and set down_revision accordingly
revision = "20251008_top15_tables"
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        "top15_holdings",
        sa.Column("symbol", sa.Text(), primary_key=True),
        sa.Column("entry_date", sa.Date(), nullable=False),
    )

    op.create_table(
        "top15_trade_intents",
        sa.Column("id", sa.BigInteger().with_variant(sa.BigInteger, "postgresql"), primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("side", sa.Text(), nullable=False),
        sa.Column("suggested_weight", sa.Float(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
    )

def downgrade() -> None:
    op.drop_table("top15_trade_intents")
    op.drop_table("top15_holdings")
