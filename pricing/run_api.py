#!/usr/bin/env python
"""Simple runner for the FastAPI application."""

if __name__ == "__main__":
    import uvicorn
    from pathlib import Path

    # Ensure we're in the right directory
    import os
    os.chdir(Path(__file__).resolve().parent)
    
    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
