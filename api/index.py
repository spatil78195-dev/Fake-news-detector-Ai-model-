"""
Vercel serverless entry point for the Flask application.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("VERCEL", "1")

from app import app  # noqa: E402
