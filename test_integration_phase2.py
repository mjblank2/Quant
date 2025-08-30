"""
Simple integration test to verify Phase 2 infrastructure components can be imported and configured.
"""

import os
import sys

# Mock DATABASE_URL to allow imports
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"


def test_imports():
    """Test that all new modules can be imported."""
    try:
        # Test validation module import
        from data.validation import ValidationResult

        print("✅ ValidationResult import successful")

        # Test ValidationResult functionality
        result = ValidationResult()
        result.add_warning("Test warning")
        result.add_error("Test error")
        result.add_metric("test_metric", 42.0)

        assert not result.passed  # Should fail due to error
        assert len(result.warnings) == 1
        assert len(result.errors) == 1
        assert result.metrics["test_metric"] == 42.0
        print("✅ ValidationResult functionality test passed")

    except Exception as e:
        print(f"❌ Validation module test failed: {e}")
        return False

    try:
        # Test configuration additions
        from config import ENABLE_TIMESCALEDB, ENABLE_DATA_VALIDATION

        print(
            f"✅ Configuration import successful: TimescaleDB={ENABLE_TIMESCALEDB}, Validation={ENABLE_DATA_VALIDATION}"
        )

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

    try:
        # Test database model additions
        from db import DataValidationLog, DataLineage

        print("✅ New database models import successful")

        # Check that models have expected attributes
        validation_table = DataValidationLog.__table__
        expected_columns = {
            "id",
            "run_timestamp",
            "validation_type",
            "status",
            "message",
            "metrics",
            "affected_symbols",
        }
        actual_columns = {col.name for col in validation_table.columns}

        if expected_columns.issubset(actual_columns):
            print("✅ DataValidationLog table structure correct")
        else:
            missing = expected_columns - actual_columns
            print(f"❌ DataValidationLog missing columns: {missing}")
            return False

        lineage_table = DataLineage.__table__
        expected_lineage_columns = {
            "id",
            "table_name",
            "symbol",
            "data_date",
            "ingestion_timestamp",
            "source",
        }
        actual_lineage_columns = {col.name for col in lineage_table.columns}

        if expected_lineage_columns.issubset(actual_lineage_columns):
            print("✅ DataLineage table structure correct")
        else:
            missing = expected_lineage_columns - actual_lineage_columns
            print(f"❌ DataLineage missing columns: {missing}")
            return False

    except Exception as e:
        print(f"❌ Database model test failed: {e}")
        return False

    return True


def test_enhanced_pipeline():
    """Test enhanced pipeline can be imported."""
    try:
        from run_pipeline import main

        print("✅ Enhanced pipeline import successful")

        # Check that the main function signature is updated
        import inspect

        sig = inspect.signature(main)
        if "sync_broker" in sig.parameters:
            print("✅ Pipeline function signature correct")
        else:
            print("❌ Pipeline function missing expected parameters")
            return False

    except Exception as e:
        print(f"❌ Enhanced pipeline test failed: {e}")
        return False

    return True


def test_cli_structure():
    """Test CLI tool structure."""
    try:
        # Check that CLI file exists and is executable
        cli_path = "scripts/data_infra_cli.py"
        if os.path.exists(cli_path):
            print("✅ CLI tool exists")

            # Check if it has executable permissions
            if os.access(cli_path, os.X_OK):
                print("✅ CLI tool is executable")
            else:
                print("⚠️  CLI tool exists but not executable")
        else:
            print("❌ CLI tool not found")
            return False

    except Exception as e:
        print(f"❌ CLI structure test failed: {e}")
        return False

    return True


def test_migration_file():
    """Test migration file exists and has correct structure."""
    try:
        migration_file = "alembic/versions/20250827_10_bitemporal_timescale.py"
        if os.path.exists(migration_file):
            print("✅ Migration file exists")

            # Read migration file and check for key components
            with open(migration_file, "r") as f:
                content = f.read()

            required_elements = [
                "knowledge_date",
                "data_validation_log",
                "data_lineage",
                "ix_fundamentals_bitemporal",
                "ix_shares_outstanding_bitemporal",
            ]

            missing_elements = []
            for element in required_elements:
                if element not in content:
                    missing_elements.append(element)

            if not missing_elements:
                print("✅ Migration file contains all required elements")
            else:
                print(f"❌ Migration file missing elements: {missing_elements}")
                return False
        else:
            print("❌ Migration file not found")
            return False

    except Exception as e:
        print(f"❌ Migration file test failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("Phase 2 Data Infrastructure Integration Test")
    print("=" * 50)

    tests = [
        ("Module Imports", test_imports),
        ("Enhanced Pipeline", test_enhanced_pipeline),
        ("CLI Structure", test_cli_structure),
        ("Migration File", test_migration_file),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"💥 {test_name} test ERROR: {e}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Phase 2 infrastructure tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
