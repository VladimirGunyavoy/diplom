"""
Simple runner for player_zoom with auto-restart on file changes
Usage: python run.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run watcher
from watcher import main as watcher_main

if __name__ == "__main__":
    watcher_main()
