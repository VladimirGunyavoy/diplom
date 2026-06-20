"""
Simple auto-restart wrapper for player_zoom
Restarts main.py after normal exit, stops on errors
"""

import os
import subprocess
import sys

# Path to player_zoom root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'main.py')


def main():
    """Main loop - restart on normal exit, stop on error"""
    print("=" * 60)
    print("PLAYER ZOOM AUTO-RESTART")
    print("=" * 60)
    print(f"Main script: {MAIN_SCRIPT}")
    print("Behavior:")
    print("  - Normal exit (code 0) -> auto-restart")
    print("  - Error exit (code != 0) -> stop")
    print("=" * 60)

    if not os.path.exists(MAIN_SCRIPT):
        print(f"ERROR: main.py not found at {MAIN_SCRIPT}")
        sys.exit(1)

    restart_count = 0

    try:
        while True:
            restart_count += 1
            print("\n" + "=" * 60)
            print(f"[START #{restart_count}] Running main.py")
            print("=" * 60)

            # Run main.py
            process = subprocess.Popen(
                [sys.executable, "-u", MAIN_SCRIPT],
                cwd=PROJECT_ROOT
            )

            # Wait for completion
            exit_code = process.wait()

            print("\n" + "=" * 60)
            if exit_code == 0:
                print(f"[EXIT] Normal exit (code {exit_code})")
                print("[INFO] Restarting in 1 second...")
                print("=" * 60)
                import time
                time.sleep(1)
            else:
                print(f"[ERROR] Exit with error code {exit_code}")
                print("[STOP] Watcher stopped - fix errors and restart manually")
                print("=" * 60)
                break

    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Ctrl+C pressed")
        print("[STOP] Watcher stopped")
    finally:
        print("\n[EXIT] Watcher finished")


if __name__ == "__main__":
    main()
