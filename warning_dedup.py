"""
Warning Deduplication Module

Provides intelligent warning rate limiting and deduplication to reduce log spam
while maintaining important error visibility.
"""
from __future__ import annotations
import logging
import time
from typing import Dict, Set, Optional
from collections import defaultdict
from threading import Lock

log = logging.getLogger(__name__)

class WarningDeduplicator:
    """
    Rate-limited warning system that prevents spam while preserving important alerts.
    """
    
    def __init__(self):
        self._warning_cache: Dict[str, float] = {}  # warning_key -> last_logged_time
        self._warning_counts: Dict[str, int] = defaultdict(int)  # warning_key -> count
        self._lock = Lock()
        
        # Different rate limits for different types of warnings
        self._rate_limits = {
            'adj_close': 300,  # 5 minutes for adjusted close warnings
            'cardinality': 60,  # 1 minute for cardinality violations
            'default': 30,     # 30 seconds for other warnings
            'critical': 5,     # 5 seconds for critical warnings
        }
    
    def should_log_warning(self, warning_key: str, warning_type: str = 'default') -> bool:
        """
        Check if a warning should be logged based on rate limiting.
        
        Args:
            warning_key: Unique identifier for the warning
            warning_type: Type of warning for rate limit classification
            
        Returns:
            True if warning should be logged, False if rate limited
        """
        with self._lock:
            current_time = time.time()
            rate_limit = self._rate_limits.get(warning_type, self._rate_limits['default'])
            
            last_logged = self._warning_cache.get(warning_key, 0)
            self._warning_counts[warning_key] += 1
            
            if current_time - last_logged >= rate_limit:
                self._warning_cache[warning_key] = current_time
                return True
            
            return False
    
    def log_warning_with_dedup(self, logger: logging.Logger, warning_key: str, 
                              message: str, warning_type: str = 'default') -> None:
        """
        Log a warning with deduplication.
        
        Args:
            logger: Logger instance to use
            warning_key: Unique identifier for the warning
            message: Warning message
            warning_type: Type of warning for rate limit classification
        """
        if self.should_log_warning(warning_key, warning_type):
            count = self._warning_counts[warning_key]
            if count > 1:
                message += f" (occurred {count} times since last log)"
            logger.warning(message)
    
    def log_debug_with_dedup(self, logger: logging.Logger, warning_key: str,
                            message: str, warning_type: str = 'default') -> None:
        """
        Log a debug message with deduplication.
        
        Args:
            logger: Logger instance to use
            warning_key: Unique identifier for the warning
            message: Debug message
            warning_type: Type of warning for rate limit classification
        """
        if self.should_log_warning(warning_key, warning_type):
            count = self._warning_counts[warning_key]
            if count > 1:
                message += f" (occurred {count} times since last log)"
            logger.debug(message)
    
    def get_warning_stats(self) -> Dict[str, int]:
        """Get statistics about warning counts."""
        with self._lock:
            return dict(self._warning_counts)
    
    def clear_cache(self) -> None:
        """Clear the warning cache (useful for testing)."""
        with self._lock:
            self._warning_cache.clear()
            self._warning_counts.clear()

# Global instance for use across the application
_global_deduplicator = WarningDeduplicator()

def warn_once(logger: logging.Logger, warning_key: str, message: str, 
              warning_type: str = 'default') -> None:
    """
    Convenience function to log a warning with deduplication.
    
    Args:
        logger: Logger instance
        warning_key: Unique identifier for the warning
        message: Warning message
        warning_type: Type of warning for rate limiting
    """
    _global_deduplicator.log_warning_with_dedup(logger, warning_key, message, warning_type)

def debug_once(logger: logging.Logger, warning_key: str, message: str,
               warning_type: str = 'default') -> None:
    """
    Convenience function to log a debug message with deduplication.
    
    Args:
        logger: Logger instance
        warning_key: Unique identifier for the warning
        message: Debug message
        warning_type: Type of warning for rate limiting
    """
    _global_deduplicator.log_debug_with_dedup(logger, warning_key, message, warning_type)

def get_warning_stats() -> Dict[str, int]:
    """Get global warning statistics."""
    return _global_deduplicator.get_warning_stats()

def clear_warning_cache() -> None:
    """Clear the global warning cache."""
    _global_deduplicator.clear_cache()

# Specific helper functions for common warning types
def warn_adj_close_missing(logger: logging.Logger, table_name: str) -> None:
    """Log a rate-limited warning about missing adj_close column."""
    warning_key = f"adj_close_missing_{table_name}"
    message = f"Dropping adj_close column (not present in {table_name})"
    warn_once(logger, warning_key, message, 'adj_close')

def warn_cardinality_violation(logger: logging.Logger, table_name: str, removed_count: int, 
                              total_count: int) -> None:
    """Log a rate-limited warning about cardinality violations."""
    warning_key = f"cardinality_{table_name}"
    percentage = (removed_count / total_count * 100) if total_count > 0 else 0
    
    if percentage > 50:  # Significant duplication
        message = f"Removed {removed_count} duplicate rows during retry (significant duplication detected)"
        warn_once(logger, warning_key, message, 'cardinality')
    else:
        message = f"Removed {removed_count} duplicate rows during retry to prevent CardinalityViolation"
        debug_once(logger, warning_key, message, 'cardinality')

def warn_column_mismatch(logger: logging.Logger, table_name: str, dropped_columns: list) -> None:
    """Log a rate-limited warning about column mismatches."""
    warning_key = f"column_mismatch_{table_name}_{hash(tuple(sorted(dropped_columns)))}"
    message = f"Dropping columns not present in {table_name}: {dropped_columns}"
    warn_once(logger, warning_key, message, 'default')