# Alembic Migration Best Practices

## PostgreSQL Transaction Handling

When writing Alembic migrations that may perform risky operations (complex SQL, extension management, data backfills), follow these best practices to prevent transaction corruption:

### Use Savepoints for Risky Operations

```python
def upgrade() -> None:
    bind = op.get_bind()
    
    # For operations that might fail in PostgreSQL
    if bind.dialect.name == "postgresql":
        try:
            with bind.begin_nested():  # Creates a savepoint
                bind.execute(sa.text("RISKY SQL OPERATION"))
        except Exception as e:
            print(f"Warning: Operation failed: {e}")
            # Savepoint automatically rolls back, main transaction continues
    else:
        # Simpler handling for other databases
        try:
            bind.execute(sa.text("RISKY SQL OPERATION"))
        except Exception as e:
            print(f"Warning: Operation failed: {e}")
```

### Common Scenarios Requiring Savepoints

1. **Data backfills with complex SQL**
2. **Extension creation** (CREATE EXTENSION)
3. **TimescaleDB hypertable operations**
4. **Dynamic SQL operations**
5. **Operations on tables that may not exist**

### Error Symptoms

If you see:
```
psycopg.errors.InFailedSqlTransaction: current transaction is aborted, commands ignored until end of transaction block
```

This indicates that an earlier operation in the transaction failed, and PostgreSQL has marked the entire transaction as failed. Using savepoints prevents this from affecting the main migration transaction.

### Testing Migrations

Always test migrations with both upgrade and downgrade:

```bash
# Test upgrade
alembic upgrade heads

# Test downgrade 
alembic downgrade <previous_revision>

# Test re-upgrade
alembic upgrade heads
```

Use the provided `test_migration_fix.py` script as a template for automated migration testing.