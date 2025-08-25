from __future__ import annotations
from datetime import date
from data.universe_history import snapshot_universe
from hedges.overlays import propose_overlays, OverlaySpec
from db import create_tables

def main():
    create_tables()
    n = snapshot_universe()
    print(f"Universe snapshot rows: {n}")
    # propose overlays for today on IWM
    today = date.today()
    propose_overlays(today, equity=1_000_000.0, spec=OverlaySpec())

if __name__ == "__main__":
    main()