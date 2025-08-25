from __future__ import annotations
import argparse, logging
from datetime import date
from data.universe_history import take_snapshot
from hedges.overlays import propose_collars

logging.basicConfig(level=logging.INFO, format='%(asctime)s - ADDONS - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', action='store_true', help='Take universe snapshot (today).')
    p.add_argument('--overlays', action='store_true', help='Propose book-level option overlays.')
    args = p.parse_args()
    if args.snapshot:
        n = take_snapshot(date.today())
        log.info(f'Took universe snapshot (rows written: {n}).')
    if args.overlays:
        m = propose_collars(date.today())
        log.info(f'Proposed {m} overlays.')

if __name__ == "__main__":
    main()
