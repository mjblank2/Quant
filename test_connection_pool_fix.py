import os
import pandas as pd
from sqlalchemy import Column, Integer, create_engine, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import ProgrammingError

# Ensure DATABASE_URL is set before importing db
os.environ.setdefault("DATABASE_URL", "sqlite://")
import db

def test_upsert_dataframe_reuses_connection(monkeypatch):
    # Create engine with a tiny connection pool
    engine = create_engine("sqlite://", poolclass=QueuePool, pool_size=1, max_overflow=0, pool_timeout=1)
    monkeypatch.setattr(db, "engine", engine)

    Base = declarative_base()

    class TestTable(Base):
        __tablename__ = "test_table"
        id = Column(Integer, primary_key=True)
        value = Column(Integer)

    Base.metadata.create_all(engine)

    df = pd.DataFrame({"id": [1, 2], "value": [10, 20]})

    with engine.begin() as conn:
        orig_execute = conn.execute
        state = {"first": True}

        def flaky(stmt, *args, **kwargs):
            if state["first"]:
                state["first"] = False
                raise ProgrammingError("CardinalityViolation", {}, None)
            return orig_execute(stmt, *args, **kwargs)

        conn.execute = flaky
        db.upsert_dataframe(df, TestTable, ["id"], conn=conn)

    # Verify rows inserted
    with engine.connect() as check_conn:
        count = check_conn.execute(text("select count(*) from test_table")).scalar()
    assert count == 2

    # Ensure no connections are left checked out
    assert engine.pool.checkedout() == 0
