from __future__ import annotations
import os
from sqlalchemy import create_engine, text

REQUIRED_COLS = {
    "market_cap": "DOUBLE PRECISION",
    "shares_outstanding": "DOUBLE PRECISION",
    "adv_usd_20": "DOUBLE PRECISION",
    "adv_usd_10": "DOUBLE PRECISION",
    "adv_usd_5": "DOUBLE PRECISION"
}

def main():
    db = os.getenv("DATABASE_URL")
    if not db:
        raise SystemExit("DATABASE_URL not set")
    if db.startswith("postgres://"):
        db = db.replace("postgres://","postgresql+psycopg://",1)
    eng = create_engine(db)
    with eng.begin() as c:
        existing = {
            r[0] for r in c.execute(text("""
              SELECT column_name FROM information_schema.columns
              WHERE table_name='universe'
            """)).fetchall()
        }
        for col, ddl_type in REQUIRED_COLS.items():
            if col not in existing:
                print(f"Adding column universe.{col}")
                c.execute(text(f"ALTER TABLE universe ADD COLUMN {col} {ddl_type}"))
            else:
                print(f"Column {col} already exists")
    print("Universe columns ensured.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
