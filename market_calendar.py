"""
Market Calendar Module

Provides market calendar functionality to handle non-trading days,
holidays, and market hours for pipeline robustness.
"""
from __future__ import annotations
import logging
from datetime import date, datetime, timedelta
from typing import List, Set
import calendar

log = logging.getLogger(__name__)

# US Market holidays (basic set - can be extended)
US_MARKET_HOLIDAYS_2024 = {
    date(2024, 1, 1),   # New Year's Day
    date(2024, 1, 15),  # Martin Luther King Jr. Day
    date(2024, 2, 19),  # Presidents' Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),   # Independence Day
    date(2024, 9, 2),   # Labor Day
    date(2024, 11, 28), # Thanksgiving
    date(2024, 12, 25), # Christmas
}

US_MARKET_HOLIDAYS_2025 = {
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # Martin Luther King Jr. Day
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
}

def is_market_day(check_date: date) -> bool:
    """
    Check if a given date is a trading day (weekday and not a holiday).
    
    Args:
        check_date: Date to check
        
    Returns:
        True if it's a trading day, False otherwise
    """
    # Check if it's a weekend
    if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check if it's a known holiday
    if check_date in US_MARKET_HOLIDAYS_2024 or check_date in US_MARKET_HOLIDAYS_2025:
        return False
    
    return True

def get_next_market_day(from_date: date = None) -> date:
    """
    Get the next trading day from the given date.
    
    Args:
        from_date: Starting date (defaults to today)
        
    Returns:
        Next trading day
    """
    if from_date is None:
        from_date = date.today()
    
    check_date = from_date
    while not is_market_day(check_date):
        check_date += timedelta(days=1)
    
    return check_date

def get_previous_market_day(from_date: date = None) -> date:
    """
    Get the previous trading day from the given date.
    
    Args:
        from_date: Starting date (defaults to today)
        
    Returns:
        Previous trading day
    """
    if from_date is None:
        from_date = date.today()
    
    check_date = from_date - timedelta(days=1)
    while not is_market_day(check_date):
        check_date -= timedelta(days=1)
    
    return check_date

def get_market_days_between(start_date: date, end_date: date) -> List[date]:
    """
    Get all trading days between two dates (inclusive).
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of trading days
    """
    market_days = []
    current_date = start_date
    
    while current_date <= end_date:
        if is_market_day(current_date):
            market_days.append(current_date)
        current_date += timedelta(days=1)
    
    return market_days

def should_run_pipeline(target_date: date = None) -> tuple[bool, str]:
    """
    Determine if the pipeline should run for a given date.
    
    Args:
        target_date: Date to check (defaults to today)
        
    Returns:
        Tuple of (should_run: bool, reason: str)
    """
    if target_date is None:
        target_date = date.today()
    
    if not is_market_day(target_date):
        if target_date.weekday() >= 5:
            reason = f"{target_date} is a weekend (no market data)"
        else:
            reason = f"{target_date} is a market holiday (no market data)"
        return False, reason
    
    # Check if we're running too early (before market close)
    now = datetime.now()
    if target_date == date.today() and now.hour < 16:  # Before 4 PM ET
        return False, f"Running too early - market not closed yet (current time: {now.strftime('%H:%M')})"
    
    return True, f"{target_date} is a valid trading day"

def get_market_calendar_info(target_date: date = None) -> dict:
    """
    Get comprehensive market calendar information for a date.
    
    Args:
        target_date: Date to analyze (defaults to today)
        
    Returns:
        Dictionary with calendar information
    """
    if target_date is None:
        target_date = date.today()
    
    info = {
        'date': target_date,
        'is_market_day': is_market_day(target_date),
        'weekday': calendar.day_name[target_date.weekday()],
        'is_weekend': target_date.weekday() >= 5,
        'is_holiday': target_date in US_MARKET_HOLIDAYS_2024 or target_date in US_MARKET_HOLIDAYS_2025,
        'next_market_day': get_next_market_day(target_date),
        'previous_market_day': get_previous_market_day(target_date),
        'should_run_pipeline': should_run_pipeline(target_date)
    }
    
    return info

def log_market_calendar_status(target_date: date = None) -> None:
    """
    Log market calendar status for debugging and monitoring.
    
    Args:
        target_date: Date to analyze (defaults to today)
    """
    info = get_market_calendar_info(target_date)
    
    log.info(f"📅 Market Calendar Status for {info['date']} ({info['weekday']}):")
    log.info(f"   Market Day: {info['is_market_day']}")
    log.info(f"   Weekend: {info['is_weekend']}")
    log.info(f"   Holiday: {info['is_holiday']}")
    log.info(f"   Previous Market Day: {info['previous_market_day']}")
    log.info(f"   Next Market Day: {info['next_market_day']}")
    
    should_run, reason = info['should_run_pipeline']
    log.info(f"   Pipeline Should Run: {should_run} ({reason})")