# Fix for Missing market_cap Column

## Problem
The `data/universe.py` script fails with the following error:
```
psycopg.errors.UndefinedColumn: column "market_cap" of relation "universe" does not exist
```

## Root Cause
The database was created using the `create_tables()` function from `db.py` instead of running proper alembic migrations. This can happen when:
1. The database was created before the `market_cap` column was added to the `Universe` model
2. The database was initialized using `create_tables()` without running migrations
3. There was a mismatch between the model definition and the actual database schema

## Solution
A new migration has been created to fix this issue:
- `alembic/versions/20250827_12_add_universe_market_cap_simple.py`

### To apply the fix:

1. **Run the migration:**
   ```bash
   alembic upgrade head
   ```

2. **Or use the data infrastructure CLI:**
   ```bash
   python scripts/data_infra_cli.py migrate
   ```

3. **Manual approach (if migrations fail):**
   ```sql
   ALTER TABLE universe ADD COLUMN market_cap FLOAT;
   ```

## Migration Details
The migration is designed to be safe and idempotent:
- It attempts to add the `market_cap` column
- If the column already exists, it ignores the error
- It works with both PostgreSQL and SQLite databases

## Verification
After applying the fix, you can verify it works by running:
```bash
python -m data.universe
```

This should now complete successfully without the column error.

## Prevention
To avoid this issue in the future:
1. Always use `alembic upgrade head` to create/update database schema
2. Avoid using `create_tables()` in production environments
3. Keep the `Universe` model in `db.py` in sync with alembic migrations