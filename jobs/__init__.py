"""
jobs package for worker entrypoints.

This package exposes the `worker` module so that process managers (such as
Render or Celery) can import `jobs.worker` as an entrypoint. Without an
``__init__.py`` file, Python would treat the ``jobs`` directory as a
namespace package and module resolution for `jobs.worker` would fail.
"""
