"""
Production Server Starter for LRP Explainability Studio
========================================================
Usage:
    python start_server.py
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("  Starting LRP Explainability Production Server")
    print("  Local URL:    http://127.0.0.1:8000")
    print("  API Docs:     http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
