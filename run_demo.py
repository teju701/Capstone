"""
run_demo.py
───────────
Cross-platform 1-click launcher for the Multi-Task Perception Cockpit.
Works on Windows, macOS, and Linux across both GPU and CPU environments.
"""

import os
import sys
import webbrowser
import threading
import time

def open_browser():
    time.sleep(1.8)
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    # Start browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run the server
    import uvicorn
    from app.server import app
    port = int(os.environ.get("PORT", 8000))
    print("\n" + "=" * 55)
    print("  Autonomous Driving Multi-Task Perception Cockpit")
    print(f"  Live at: http://localhost:{port}")
    print("=" * 55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
