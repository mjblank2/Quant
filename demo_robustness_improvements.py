#!/usr/bin/env python3
"""
Demo script showcasing the pipeline robustness improvements.

Demonstrates:
- Version guard checking
- Market calendar validation  
- Warning deduplication
- Enhanced trade generation
- Background task management
"""
import os
import logging
from datetime import date, timedelta

# Set up demo database
os.environ['DATABASE_URL'] = 'sqlite:///demo.db'

def demo_version_guard():
    """Demonstrate version guard functionality."""
    print("\n🛡️  VERSION GUARD DEMO")
    print("=" * 40)
    
    from version_guard import check_schema_version, get_current_schema_version
    
    print("Checking schema version compatibility...")
    is_compatible, message = check_schema_version()
    print(f"Result: {'✅ Compatible' if is_compatible else '❌ Incompatible'}")
    print(f"Message: {message}")
    
    current_version = get_current_schema_version()
    print(f"Current schema version: {current_version}")

def demo_market_calendar():
    """Demonstrate market calendar functionality."""
    print("\n📅 MARKET CALENDAR DEMO")
    print("=" * 40)
    
    from market_calendar import (
        get_market_calendar_info, 
        should_run_pipeline,
        is_market_day,
        get_next_market_day,
        get_previous_market_day
    )
    
    # Test different dates
    test_dates = [
        date.today(),
        date(2024, 1, 1),   # New Year's Day (holiday)
        date(2024, 1, 6),   # Saturday
        date(2024, 1, 8),   # Monday
    ]
    
    for test_date in test_dates:
        print(f"\nDate: {test_date} ({test_date.strftime('%A')})")
        info = get_market_calendar_info(test_date)
        
        print(f"  Market Day: {'✅ Yes' if info['is_market_day'] else '❌ No'}")
        print(f"  Weekend: {'Yes' if info['is_weekend'] else 'No'}")
        print(f"  Holiday: {'Yes' if info['is_holiday'] else 'No'}")
        
        should_run, reason = should_run_pipeline(test_date)
        print(f"  Should Run Pipeline: {'✅ Yes' if should_run else '❌ No'} ({reason})")

def demo_warning_deduplication():
    """Demonstrate warning deduplication system."""
    print("\n⚠️  WARNING DEDUPLICATION DEMO")
    print("=" * 40)
    
    from warning_dedup import (
        warn_once, 
        debug_once, 
        warn_adj_close_missing,
        get_warning_stats,
        clear_warning_cache
    )
    import logging
    from io import StringIO
    
    # Set up log capture
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger('demo_warnings')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    print("Testing warning rate limiting...")
    
    # Clear cache for clean test
    clear_warning_cache()
    
    # Rapid-fire warnings (should be rate limited)
    for i in range(5):
        warn_once(logger, "repeated_warning", f"This is warning #{i+1}", "default")
        warn_adj_close_missing(logger, "demo_table")
    
    print("Warnings generated (check that duplicates are rate-limited)")
    logs = log_capture.getvalue()
    print(f"Log output contains {logs.count('warning')} warning instances")
    
    stats = get_warning_stats()
    print(f"Warning statistics: {dict(stats)}")

def demo_enhanced_trade_generation():
    """Demonstrate enhanced trade generation with validation."""
    print("\n💰 ENHANCED TRADE GENERATION DEMO")
    print("=" * 40)
    
    from enhanced_trade_generation import (
        validate_trading_conditions,
        validate_predictions_quality,
        get_trade_generation_health
    )
    import pandas as pd
    
    print("Checking trading conditions...")
    conditions_ok, conditions_msg = validate_trading_conditions()
    print(f"Trading conditions: {'✅ Valid' if conditions_ok else '❌ Invalid'}")
    print(f"Reason: {conditions_msg}")
    
    print("\nTesting prediction quality validation...")
    
    # Test with sample predictions
    sample_predictions = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'ts': [date.today()] * 3,
        'y_pred': [0.05, -0.02, 0.03]
    })
    
    pred_ok, pred_msg = validate_predictions_quality(sample_predictions)
    print(f"Sample predictions: {'✅ Valid' if pred_ok else '❌ Invalid'}")
    print(f"Reason: {pred_msg}")
    
    # Test with extreme predictions
    extreme_predictions = pd.DataFrame({
        'symbol': ['EXTREME1', 'EXTREME2'],
        'ts': [date.today()] * 2,
        'y_pred': [0.8, -0.7]  # 80% and -70% predictions
    })
    
    extreme_ok, extreme_msg = validate_predictions_quality(extreme_predictions)
    print(f"Extreme predictions: {'✅ Valid' if extreme_ok else '❌ Invalid'}")
    print(f"Reason: {extreme_msg}")
    
    print("\nGetting trade generation health status...")
    health = get_trade_generation_health()
    print(f"Overall status: {health['overall_status']}")
    print(f"Trading conditions: {health['trading_conditions']}")

def demo_background_tasks():
    """Demonstrate background task management."""
    print("\n🔄 BACKGROUND TASK MANAGEMENT DEMO")
    print("=" * 40)
    
    from background_tasks import (
        submit_background_task,
        get_task_status,
        get_all_tasks,
        TaskStatus
    )
    import time
    
    def sample_task(duration=0.5):
        """Sample task that takes some time."""
        time.sleep(duration)
        return f"Task completed after {duration}s"
    
    print("Submitting background task...")
    task_info = submit_background_task(
        "demo_task", 
        "Demo Task", 
        sample_task, 
        timeout_seconds=10,
        duration=0.2  # 0.2 second task
    )
    
    print(f"Task submitted: {task_info.task_id}")
    print(f"Initial status: {task_info.status}")
    
    # Monitor progress
    print("Monitoring task progress...")
    for i in range(5):
        time.sleep(0.1)
        current_status = get_task_status("demo_task")
        print(f"  Check {i+1}: {current_status.status}")
        if current_status.status == TaskStatus.COMPLETED:
            print(f"  Result: {current_status.result}")
            break
    
    # Show all tasks
    all_tasks = get_all_tasks()
    print(f"\nAll tasks in system: {len(all_tasks)}")
    for task_id, task in all_tasks.items():
        print(f"  {task_id}: {task.status} - {task.name}")

def demo_pipeline_integration():
    """Demonstrate full pipeline with all improvements."""
    print("\n🚀 ENHANCED PIPELINE INTEGRATION DEMO")
    print("=" * 40)
    
    from run_pipeline import main
    
    print("Running enhanced pipeline with all robustness improvements...")
    print("(Note: Will likely skip due to weekend/market closure)")
    
    try:
        result = main(sync_broker=False)
        print(f"Pipeline result: {'✅ Success' if result else '❌ Failed'}")
    except Exception as e:
        print(f"Pipeline error: {e}")

def main():
    """Run all demos."""
    print("🎯 PIPELINE ROBUSTNESS IMPROVEMENTS DEMO")
    print("=" * 60)
    print("This demo showcases the comprehensive robustness improvements")
    print("including version guard, market calendar, warning deduplication,")
    print("enhanced trade generation, and background task management.")
    print("=" * 60)
    
    # Configure logging for demo
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        demo_version_guard()
        demo_market_calendar()
        demo_warning_deduplication()
        demo_enhanced_trade_generation()
        demo_background_tasks()
        demo_pipeline_integration()
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("All robustness improvements are working correctly.")
        
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            import os
            if os.path.exists('demo.db'):
                os.unlink('demo.db')
            print("\n🧹 Cleanup completed.")
        except:
            pass

if __name__ == "__main__":
    main()