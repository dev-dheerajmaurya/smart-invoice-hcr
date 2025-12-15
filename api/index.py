import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ocr_service.app import app

# Vercel expects the FastAPI app directly
handler = app
