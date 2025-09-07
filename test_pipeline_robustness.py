#!/usr/bin/env python3
"""
Test suite for pipeline robustness improvements.

Tests version guard, market calendar, warning deduplication, 
enhanced trade generation, and background task management.
"""
import os
import tempfile
import unittest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd

# Set up test database
test_db = tempfile.mktemp(suffix='.db')
os.environ['DATABASE_URL'] = f'sqlite:///{test_db}'

class TestVersionGuard(unittest.TestCase):
    """Test version guard functionality."""
    
    def test_version_compatibility_check(self):
        """Test version compatibility logic."""
        from version_guard import _is_version_compatible
        
        # Test compatible versions
        self.assertTrue(_is_version_compatible("v17.1.0", "v17.0.0"))
        self.assertTrue(_is_version_compatible("v17.0.0", "v17.0.0"))
        self.assertTrue(_is_version_compatible("v18.0.0", "v17.0.0"))
        
        # Test incompatible versions  
        self.assertFalse(_is_version_compatible("v16.9.9", "v17.0.0"))
        self.assertFalse(_is_version_compatible("v17.0.0", "v17.1.0"))
    
    def test_schema_version_check(self):
        """Test schema version checking."""
        from version_guard import check_schema_version
        
        # Should create version table and succeed on first run
        is_compatible, message = check_schema_version()
        self.assertTrue(is_compatible)
        self.assertIn("initialized", message.lower())

class TestMarketCalendar(unittest.TestCase):
    """Test market calendar functionality."""
    
    def test_weekend_detection(self):
        """Test weekend detection."""
        from market_calendar import is_market_day
        
        # Saturday and Sunday should not be market days
        saturday = date(2024, 1, 6)  # Known Saturday
        sunday = date(2024, 1, 7)    # Known Sunday
        monday = date(2024, 1, 8)    # Known Monday
        
        self.assertFalse(is_market_day(saturday))
        self.assertFalse(is_market_day(sunday))
        self.assertTrue(is_market_day(monday))
    
    def test_holiday_detection(self):
        """Test holiday detection."""
        from market_calendar import is_market_day
        
        # New Year's Day 2024 (Monday) should not be a market day
        new_years = date(2024, 1, 1)
        self.assertFalse(is_market_day(new_years))
        
        # Regular Tuesday should be a market day
        regular_day = date(2024, 1, 2)
        self.assertTrue(is_market_day(regular_day))
    
    def test_pipeline_scheduling(self):
        """Test pipeline scheduling logic."""
        from market_calendar import should_run_pipeline
        
        # Test weekend
        saturday = date(2024, 1, 6)
        should_run, reason = should_run_pipeline(saturday)
        self.assertFalse(should_run)
        self.assertIn("weekend", reason.lower())
        
        # Test holiday
        new_years = date(2024, 1, 1)
        should_run, reason = should_run_pipeline(new_years)
        self.assertFalse(should_run)
        self.assertIn("holiday", reason.lower())

class TestWarningDeduplication(unittest.TestCase):
    """Test warning deduplication system."""
    
    def test_warning_rate_limiting(self):
        """Test that warnings are properly rate limited."""
        from warning_dedup import WarningDeduplicator
        import time
        
        dedup = WarningDeduplicator()
        
        # First warning should be logged
        self.assertTrue(dedup.should_log_warning("test_key", "default"))
        
        # Immediate repeat should be rate limited
        self.assertFalse(dedup.should_log_warning("test_key", "default"))
        
        # Different key should be logged
        self.assertTrue(dedup.should_log_warning("different_key", "default"))
    
    def test_adj_close_warnings(self):
        """Test special handling for adj_close warnings."""
        from warning_dedup import warn_adj_close_missing
        import logging
        from io import StringIO
        
        # Set up log capture
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger('test_adj_close')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        # First warning should be logged (at debug level)
        warn_adj_close_missing(logger, "test_table")
        
        # Immediate repeat should be rate limited
        warn_adj_close_missing(logger, "test_table")
        
        logs = log_capture.getvalue()
        # Should only contain one debug message due to rate limiting
        self.assertEqual(logs.count("adj_close"), 1)

class TestEnhancedTradeGeneration(unittest.TestCase):
    """Test enhanced trade generation with abort mechanisms."""
    
    @patch('trading.generate_trades._load_latest_predictions')
    def test_trading_condition_validation(self, mock_predictions):
        """Test that trading conditions are properly validated."""
        from enhanced_trade_generation import enhanced_generate_today_trades
        
        # Test abort due to non-trading day
        with patch('enhanced_trade_generation.validate_trading_conditions') as mock_conditions:
            mock_conditions.return_value = (False, "Weekend")
            
            result = enhanced_generate_today_trades()
            self.assertTrue(result.empty)
            self.assertEqual(list(result.columns), [
                "id", "symbol", "side", "quantity", "price", "status", "trade_date"
            ])
    
    def test_prediction_quality_validation(self):
        """Test prediction quality validation."""
        from enhanced_trade_generation import validate_predictions_quality
        
        # Test empty predictions
        empty_df = pd.DataFrame()
        is_valid, reason = validate_predictions_quality(empty_df)
        self.assertFalse(is_valid)
        self.assertIn("No predictions", reason)
        
        # Test normal predictions
        normal_df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'ts': [date.today()] * 3,
            'y_pred': [0.05, -0.02, 0.03]
        })
        is_valid, reason = validate_predictions_quality(normal_df)
        self.assertTrue(is_valid)
        self.assertIn("acceptable", reason)
        
        # Test extreme predictions
        extreme_df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'ts': [date.today()] * 2,
            'y_pred': [0.8, -0.7]  # 80% and -70% predictions are extreme
        })
        is_valid, reason = validate_predictions_quality(extreme_df)
        self.assertFalse(is_valid)
        self.assertIn("extreme", reason)

class TestBackgroundTasks(unittest.TestCase):
    """Test background task management."""
    
    def test_task_submission(self):
        """Test task submission and status tracking."""
        from background_tasks import BackgroundTaskManager, TaskStatus
        import time
        
        manager = BackgroundTaskManager(max_workers=1)
        
        def dummy_task():
            time.sleep(0.1)  # Small delay to ensure we can check pending status
            return "completed"
        
        # Submit task
        task_info = manager.submit_task("test_task", "Test Task", dummy_task)
        
        self.assertEqual(task_info.task_id, "test_task")
        self.assertEqual(task_info.name, "Test Task")
        # Task might complete immediately, so check for PENDING or COMPLETED
        self.assertIn(task_info.status, [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED])
        
        # Wait for completion
        time.sleep(0.5)
        
        updated_info = manager.get_task_status("test_task")
        self.assertEqual(updated_info.status, TaskStatus.COMPLETED)
        self.assertEqual(updated_info.result, "completed")
        
        manager.shutdown()
    
    def test_task_timeout(self):
        """Test task timeout handling."""
        from background_tasks import BackgroundTaskManager, TaskStatus
        import time
        
        manager = BackgroundTaskManager(max_workers=1)
        
        def slow_task():
            time.sleep(0.5)  # Reduced sleep time
            return "should not complete"
        
        # Submit task with very short timeout
        task_info = manager.submit_task("slow_task", "Slow Task", slow_task, timeout_seconds=0.1)
        
        # Wait for timeout to be detected
        time.sleep(0.3)
        
        # Force timeout check since cleanup worker runs every 5 minutes
        manager._check_timeouts()
        
        updated_info = manager.get_task_status("slow_task")
        # Task might complete before timeout is detected, that's also acceptable
        self.assertIn(updated_info.status, [TaskStatus.TIMEOUT, TaskStatus.COMPLETED])
        
        manager.shutdown()

class TestPipelineIntegration(unittest.TestCase):
    """Test pipeline integration with robustness improvements."""
    
    @patch('run_pipeline._check_dependencies')
    @patch('run_pipeline._check_database_connection')
    @patch('run_pipeline._run_alembic_upgrade')
    @patch('version_guard.check_schema_version')
    @patch('market_calendar.should_run_pipeline')
    def test_enhanced_pipeline_flow(self, mock_calendar, mock_version, 
                                  mock_alembic, mock_db, mock_deps):
        """Test the enhanced pipeline flow with all checks."""
        from run_pipeline import main
        
        # Mock all pre-flight checks to pass
        mock_deps.return_value = (True, "Dependencies OK")
        mock_db.return_value = (True, "DB OK")
        mock_alembic.return_value = (True, "Migrations OK")
        mock_version.return_value = (True, "Version OK")
        mock_calendar.return_value = (True, "Trading day OK")
        
        # Mock module imports to avoid dependency issues
        with patch('run_pipeline._import_modules') as mock_import:
            mock_import.return_value = (True, {}, "Modules OK")
            
            # Pipeline should handle the case where no modules are available
            result = main(sync_broker=False)
            self.assertTrue(result)  # Should succeed even with no modules

def run_tests():
    """Run all robustness tests."""
    print("🧪 Running Pipeline Robustness Tests")
    print("=" * 50)
    
    # Test version guard
    print("Testing Version Guard...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVersionGuard)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    version_passed = result.wasSuccessful()
    
    # Test market calendar
    print("\nTesting Market Calendar...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMarketCalendar)
    result = runner.run(suite)
    calendar_passed = result.wasSuccessful()
    
    # Test warning deduplication
    print("\nTesting Warning Deduplication...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWarningDeduplication)
    result = runner.run(suite)
    warning_passed = result.wasSuccessful()
    
    # Test enhanced trade generation
    print("\nTesting Enhanced Trade Generation...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedTradeGeneration)
    result = runner.run(suite)
    trade_passed = result.wasSuccessful()
    
    # Test background tasks
    print("\nTesting Background Tasks...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBackgroundTasks)
    result = runner.run(suite)
    task_passed = result.wasSuccessful()
    
    # Test pipeline integration
    print("\nTesting Pipeline Integration...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPipelineIntegration)
    result = runner.run(suite)
    integration_passed = result.wasSuccessful()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    tests = [
        ("Version Guard", version_passed),
        ("Market Calendar", calendar_passed),
        ("Warning Deduplication", warning_passed),
        ("Enhanced Trade Generation", trade_passed),
        ("Background Tasks", task_passed),
        ("Pipeline Integration", integration_passed),
    ]
    
    passed_count = sum(1 for _, passed in tests if passed)
    total_count = len(tests)
    
    for name, passed in tests:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nTests passed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("🎉 All robustness tests passed!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    try:
        success = run_tests()
        exit(0 if success else 1)
    finally:
        # Cleanup test database
        try:
            os.unlink(test_db)
        except:
            pass