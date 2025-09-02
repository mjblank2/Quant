import os
import subprocess


def test_entrypoint_api_allows_missing_database_url(tmp_path):
    env = os.environ.copy()
    env.pop('DATABASE_URL', None)
    env.update({'SERVICE': 'web', 'APP_MODE': 'api'})
    result = subprocess.run(
        ['bash', 'scripts/entrypoint.sh', 'echo', 'hello'],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'DATABASE_URL not set - running without database access' in result.stdout
    assert 'hello' in result.stdout
