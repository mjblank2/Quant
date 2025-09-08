import os
import sys
import subprocess

def test_health_api_starts_without_database():
    env = os.environ.copy()
    env.pop('DATABASE_URL', None)
    code = (
        'from fastapi.testclient import TestClient\n'
        'import health_api\n'
        'assert health_api.TASK_QUEUE_AVAILABLE is False\n'
        'client = TestClient(health_api.app)\n'
        'resp = client.get("/health")\n'
        'assert resp.status_code == 200\n'
    )
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
