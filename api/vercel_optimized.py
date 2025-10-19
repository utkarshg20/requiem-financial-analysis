"""
Vercel-optimized FastAPI application
Minimal dependencies for serverless deployment
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create minimal app
app = FastAPI(title="Requiem API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_ui():
    """Serve the main UI"""
    return FileResponse("ui/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/api/health")
async def api_health():
    """API health check"""
    return {"ok": True}

@app.post("/api/query/intelligent")
async def handle_intelligent_query():
    """Placeholder for intelligent queries"""
    return {
        "success": True,
        "message": "Vercel deployment successful! Full functionality coming soon.",
        "intent": "placeholder"
    }

# Serve static files
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="ui"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
