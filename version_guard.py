"""
Version Guard Module

Provides version checking and compatibility validation between code and database schema.
Ensures the pipeline fails fast if there's a version mismatch that could cause issues.
"""
from __future__ import annotations
import logging
from typing import Tuple
from sqlalchemy import text, create_engine
from config import DATABASE_URL

log = logging.getLogger(__name__)

# Current schema version - increment when making breaking changes
CURRENT_SCHEMA_VERSION = "v17.1.0"
MIN_COMPATIBLE_SCHEMA_VERSION = "v17.0.0"

def check_schema_version() -> Tuple[bool, str]:
    """
    Check if the database schema version is compatible with the current code.
    
    Returns:
        Tuple of (is_compatible: bool, message: str)
    """
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            # Create schema_version table if it doesn't exist
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    version VARCHAR(32) NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT single_row CHECK (id = 1)
                )
            """))
            
            # Check current version
            result = conn.execute(text("SELECT version FROM schema_version WHERE id = 1")).fetchone()
            
            if result is None:
                # First time setup - insert current version
                conn.execute(text(
                    "INSERT INTO schema_version (id, version) VALUES (1, :version)"
                ), {"version": CURRENT_SCHEMA_VERSION})
                conn.commit()
                log.info(f"Schema version initialized to {CURRENT_SCHEMA_VERSION}")
                return True, f"Schema version initialized to {CURRENT_SCHEMA_VERSION}"
            
            db_version = result[0]
            
            # Check compatibility
            if _is_version_compatible(db_version, MIN_COMPATIBLE_SCHEMA_VERSION):
                if db_version != CURRENT_SCHEMA_VERSION:
                    log.info(f"Schema version {db_version} is compatible with code {CURRENT_SCHEMA_VERSION}")
                return True, f"Schema version {db_version} is compatible (already initialized)"
            else:
                error_msg = (f"Schema version mismatch: database has {db_version}, "
                           f"but code requires >= {MIN_COMPATIBLE_SCHEMA_VERSION}")
                log.error(error_msg)
                return False, error_msg
                
    except Exception as e:
        error_msg = f"Version check failed: {str(e)}"
        log.error(error_msg)
        return False, error_msg

def update_schema_version(new_version: str) -> bool:
    """
    Update the schema version in the database.
    
    Args:
        new_version: New version string to set
        
    Returns:
        True if successful, False otherwise
    """
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE schema_version SET version = :version, updated_at = CURRENT_TIMESTAMP WHERE id = 1"
            ), {"version": new_version})
            conn.commit()
            log.info(f"Schema version updated to {new_version}")
            return True
    except Exception as e:
        log.error(f"Failed to update schema version: {e}")
        return False

def _is_version_compatible(db_version: str, min_version: str) -> bool:
    """
    Check if database version is compatible with minimum required version.
    
    Args:
        db_version: Database schema version
        min_version: Minimum required version
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        # Simple version comparison - assumes semantic versioning
        # Extract major.minor.patch numbers
        db_parts = [int(x) for x in db_version.replace('v', '').split('.')]
        min_parts = [int(x) for x in min_version.replace('v', '').split('.')]
        
        # Pad with zeros if needed
        while len(db_parts) < 3:
            db_parts.append(0)
        while len(min_parts) < 3:
            min_parts.append(0)
            
        # Compare major.minor.patch
        for i in range(3):
            if db_parts[i] > min_parts[i]:
                return True
            elif db_parts[i] < min_parts[i]:
                return False
        
        return True  # Equal versions are compatible
        
    except (ValueError, IndexError):
        log.warning(f"Cannot parse versions for comparison: {db_version} vs {min_version}")
        return False

def get_current_schema_version() -> str:
    """Get the current schema version from the database."""
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version FROM schema_version WHERE id = 1")).fetchone()
            return result[0] if result else "unknown"
    except Exception:
        return "unknown"