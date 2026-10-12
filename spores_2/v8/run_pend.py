"""
Runner for the pendulum window (main_pend.py) with auto-restart, as run.py does for the DI window
Usage: python run_pend.py
"""

import sys
import os

os.environ['V8_MAIN'] = 'main_pend.py'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.watcher import main as watcher_main

if __name__ == "__main__":
    watcher_main()
