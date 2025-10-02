"""
Module entrypoint for the quant worker.

Render’s process manager looks for a module named ``jobs.worker`` when
starting the quant-worker service.  Previously, this module did not
exist, causing an ImportError.  This file serves as a thin wrapper
around the project’s ``worker.py`` script.  It exposes a ``main``
function and, when executed as a script, invokes ``main()`` directly.
"""

from worker import main as _worker_main


def main() -> None:
    """Invoke the real worker main function."""
    _worker_main()


if __name__ == "__main__":
    main()
