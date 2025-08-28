#!/usr/bin/env python3
"""
Data Infrastructure Management CLI

Provides command-line interface for managing TimescaleDB, data validation,
and other institutional-grade data infrastructure components.

Usage:
    python scripts/data_infra_cli.py --help
    python scripts/data_infra_cli.py timescale --setup
    python scripts/data_infra_cli.py validate --full
    python scripts/data_infra_cli.py health-check
"""
from __future__ import annotations
import argparse
import logging
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.timescale import (
    setup_timescaledb, 
    get_timescaledb_info,
    is_timescaledb_available,
    enable_timescaledb_extension
)
from data.validation import run_validation_pipeline
from data.institutional_ingest import run_infrastructure_health_check
from db import create_tables

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def cmd_timescale(args):
    """Handle TimescaleDB operations."""
    if args.setup:
        log.info("Setting up TimescaleDB infrastructure")
        success = setup_timescaledb()
        if success:
            print("✅ TimescaleDB setup completed successfully")
        else:
            print("❌ TimescaleDB setup encountered errors")
            return 1
    
    elif args.info:
        log.info("Getting TimescaleDB information")
        info = get_timescaledb_info()
        print("TimescaleDB Status:")
        print(json.dumps(info, indent=2, default=str))
    
    elif args.enable:
        log.info("Enabling TimescaleDB extension")
        success = enable_timescaledb_extension()
        if success:
            print("✅ TimescaleDB extension enabled")
        else:
            print("❌ Failed to enable TimescaleDB extension")
            return 1
    
    elif args.check:
        available = is_timescaledb_available()
        if available:
            print("✅ TimescaleDB is available and ready")
        else:
            print("❌ TimescaleDB is not available")
            return 1
    
    return 0

def cmd_validate(args):
    """Handle data validation operations."""
    log.info("Running data validation pipeline")
    
    symbols = None
    if args.symbols:
        symbols = args.symbols.split(',')
    
    result = run_validation_pipeline(symbols)
    
    print(f"\nValidation Results:")
    print(f"Overall Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠️  {warning}")
    
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  ❌ {error}")
    
    if result.metrics:
        print("\nMetrics:")
        for metric, value in result.metrics.items():
            print(f"  📊 {metric}: {value}")
    
    return 0 if result.passed else 1

def cmd_health_check(args):
    """Handle infrastructure health check."""
    log.info("Running infrastructure health check")
    
    health_status = run_infrastructure_health_check()
    
    status_emoji = {
        'HEALTHY': '✅',
        'WARNING': '⚠️',
        'CRITICAL': '❌',
        'ERROR': '💥'
    }
    
    print(f"\nInfrastructure Health Check")
    print(f"Overall Status: {status_emoji.get(health_status['overall_status'], '❓')} {health_status['overall_status']}")
    
    if health_status.get('recommendations'):
        print("\nRecommendations:")
        for rec in health_status['recommendations']:
            print(f"  💡 {rec}")
    
    print("\nDetailed Checks:")
    print(json.dumps(health_status['checks'], indent=2, default=str))
    
    return 0 if health_status['overall_status'] in ['HEALTHY', 'WARNING'] else 1

def cmd_migrate(args):
    """Handle database migrations."""
    log.info("Running database migrations")
    
    try:
        # Run alembic upgrade to all heads
        import subprocess
        result = subprocess.run(['alembic', 'upgrade', 'heads'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Alembic migrations completed")
        else:
            print(f"❌ Alembic migration failed: {result.stderr}")
            return 1
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return 1
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Data Infrastructure Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s timescale --setup              Set up TimescaleDB infrastructure
  %(prog)s timescale --info               Show TimescaleDB status
  %(prog)s validate --full               Run full data validation
  %(prog)s validate --symbols AAPL,MSFT  Validate specific symbols
  %(prog)s health-check                  Run infrastructure health check
  %(prog)s migrate                       Run database migrations
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # TimescaleDB commands
    timescale_parser = subparsers.add_parser('timescale', help='TimescaleDB operations')
    timescale_group = timescale_parser.add_mutually_exclusive_group(required=True)
    timescale_group.add_argument('--setup', action='store_true', 
                                help='Set up TimescaleDB infrastructure')
    timescale_group.add_argument('--info', action='store_true',
                                help='Show TimescaleDB information')
    timescale_group.add_argument('--enable', action='store_true',
                                help='Enable TimescaleDB extension')
    timescale_group.add_argument('--check', action='store_true',
                                help='Check TimescaleDB availability')
    
    # Validation commands
    validate_parser = subparsers.add_parser('validate', help='Data validation operations')
    validate_parser.add_argument('--symbols', type=str,
                                help='Comma-separated list of symbols to validate')
    validate_parser.add_argument('--full', action='store_true',
                                help='Run full validation pipeline')
    
    # Health check command
    subparsers.add_parser('health-check', help='Run infrastructure health check')
    
    # Migration command
    subparsers.add_parser('migrate', help='Run database migrations')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate handler
    handlers = {
        'timescale': cmd_timescale,
        'validate': cmd_validate,
        'health-check': cmd_health_check,
        'migrate': cmd_migrate
    }
    
    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == '__main__':
    sys.exit(main())