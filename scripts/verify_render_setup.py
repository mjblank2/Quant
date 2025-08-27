#!/usr/bin/env python3
"""
Render Deployment Verification Script

This script verifies that all necessary components are in place for successful
deployment to Render, including configuration files, environment variables,
and application structure.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any

def check_file_exists(filepath: str, required: bool = True) -> bool:
    """Check if a file exists and report status."""
    exists = Path(filepath).exists()
    status = "✅" if exists else ("❌" if required else "⚠️")
    req_text = " (required)" if required else " (optional)"
    print(f"{status} {filepath}{req_text}")
    return exists

def check_dockerfile() -> bool:
    """Verify Dockerfile configuration."""
    print("\n📦 Checking Dockerfile...")
    if not check_file_exists("Dockerfile"):
        return False
    
    with open("Dockerfile", "r") as f:
        content = f.read()
    
    required_commands = [
        "FROM python:3.12-slim",
        "COPY requirements.txt",
        "RUN pip install",
        "ENTRYPOINT"
    ]
    
    all_good = True
    for cmd in required_commands:
        if cmd in content:
            print(f"✅ Found: {cmd}")
        else:
            print(f"❌ Missing: {cmd}")
            all_good = False
    
    return all_good

def check_render_yaml() -> bool:
    """Verify render.yaml configuration."""
    print("\n🚀 Checking render.yaml...")
    if not check_file_exists("render.yaml"):
        return False
    
    try:
        with open("render.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        services = config.get("services", [])
        print(f"✅ Found {len(services)} services defined")
        
        required_services = ["web", "worker", "redis"]
        found_services = []
        
        for service in services:
            service_type = service.get("type", "unknown")
            service_name = service.get("name", "unnamed")
            found_services.append(service_type)
            print(f"  📋 {service_type}: {service_name}")
        
        missing_services = set(required_services) - set(found_services)
        if missing_services:
            print(f"⚠️  Recommended services not found: {missing_services}")
        
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML: {e}")
        return False

def check_entrypoint() -> bool:
    """Verify entrypoint script."""
    print("\n🎯 Checking entrypoint script...")
    if not check_file_exists("scripts/entrypoint.sh"):
        return False
    
    with open("scripts/entrypoint.sh", "r") as f:
        content = f.read()
    
    required_modes = ["web", "worker", "cron"]
    all_good = True
    
    for mode in required_modes:
        if f'{mode})' in content:  # Look for case statement patterns
            print(f"✅ Supports SERVICE={mode}")
        else:
            print(f"❌ Missing SERVICE={mode} support")
            all_good = False
    
    return all_good

def check_requirements() -> bool:
    """Verify requirements files."""
    print("\n📚 Checking requirements...")
    
    main_reqs = check_file_exists("requirements.txt", required=True)
    extra_reqs = check_file_exists("requirements.extra.txt", required=False)
    
    if not main_reqs:
        return False
    
    with open("requirements.txt", "r") as f:
        reqs = f.read()
    
    critical_packages = [
        "streamlit",
        "pandas", 
        "sqlalchemy",
        "fastapi",
        "uvicorn",
        "celery",
        "redis"
    ]
    
    missing_packages = []
    for pkg in critical_packages:
        if pkg.lower() in reqs.lower():
            print(f"✅ {pkg}")
        else:
            print(f"❌ Missing: {pkg}")
            missing_packages.append(pkg)
    
    return len(missing_packages) == 0

def check_configuration_files() -> bool:
    """Check configuration and documentation files."""
    print("\n⚙️  Checking configuration files...")
    
    config_files = {
        "config.py": True,
        "alembic.ini": True,
        ".env": False,
        ".env.render.template": False,
        ".streamlit/config.toml": True,
        "DEPLOY_TO_RENDER.md": False
    }
    
    all_required = True
    for filepath, required in config_files.items():
        exists = check_file_exists(filepath, required)
        if required and not exists:
            all_required = False
    
    return all_required

def check_application_structure() -> bool:
    """Verify application module structure."""
    print("\n🏗️  Checking application structure...")
    
    required_modules = [
        "app.py",
        "db.py", 
        "health_api.py",
        "run_pipeline.py",
        "data/",
        "models/",
        "trading/",
        "tasks/"
    ]
    
    all_good = True
    for module in required_modules:
        is_dir = module.endswith("/")
        path = Path(module)
        
        if is_dir:
            exists = path.is_dir()
        else:
            exists = path.is_file()
        
        if exists:
            print(f"✅ {module}")
        else:
            print(f"❌ Missing: {module}")
            all_good = False
    
    return all_good

def check_environment_template() -> bool:
    """Check if environment template is complete."""
    print("\n🔐 Checking environment template...")
    
    if not Path(".env.render.template").exists():
        print("⚠️  No environment template found")
        return False
    
    with open(".env.render.template", "r") as f:
        template = f.read()
    
    required_vars = [
        "DATABASE_URL",
        "REDIS_URL", 
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "POLYGON_API_KEY",
        "PYTHONPATH"
    ]
    
    missing_vars = []
    for var in required_vars:
        if var in template:
            print(f"✅ {var}")
        else:
            print(f"❌ Missing: {var}")
            missing_vars.append(var)
    
    return len(missing_vars) == 0

def main():
    """Run all verification checks."""
    print("🔍 Render Deployment Verification\n")
    print("=" * 50)
    
    checks = [
        ("Dockerfile", check_dockerfile),
        ("render.yaml", check_render_yaml), 
        ("Entrypoint Script", check_entrypoint),
        ("Requirements", check_requirements),
        ("Configuration Files", check_configuration_files),
        ("Application Structure", check_application_structure),
        ("Environment Template", check_environment_template)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            results[name] = False
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, passed_check in results.items():
        status = "✅" if passed_check else "❌"
        print(f"{status} {name}")
    
    print(f"\n📈 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Ready for Render deployment.")
        print("\nNext steps:")
        print("1. Commit your changes to GitHub")
        print("2. Create services in Render dashboard")
        print("3. Set environment variables from .env.render.template")
        print("4. Deploy using the render.yaml blueprint")
        return True
    else:
        print(f"\n⚠️  {total - passed} issues found. Please fix before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)