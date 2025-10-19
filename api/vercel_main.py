"""
Vercel-compatible FastAPI application
Uses the exact same main.py but with Vercel optimizations
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Import your existing main app
from api.main import app

# Create Vercel-compatible wrapper
vercel_app = FastAPI(title="Requiem API", version="1.0.0")

# Add CORS middleware for Vercel
vercel_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
vercel_app.mount("/static", StaticFiles(directory="ui"), name="static")

@vercel_app.get("/")
async def serve_ui():
    """Serve the main UI"""
    return FileResponse("ui/index.html")

# Include all routes from your main app
for route in app.routes:
    vercel_app.routes.append(route)

# Export for Vercel
app = vercel_app