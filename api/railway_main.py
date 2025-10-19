"""
Railway-optimized main app with minimal dependencies
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uuid
import logging
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import only essential components
from workers.api_paths import StrategySpec
from workers.exceptions import ValidationError, DataError, InternalError, ExternalAPIError
from workers.engine.prompt_parser import parse_prompt, prompt_to_spec_skeleton, validate_spec_skeleton
from workers.engine.planning import plan_generator, plan_executor
from workers.engine.intelligent_analyzer import IntelligentAnalyzer
from workers.engine.talib_tool_executor import TALibToolExecutor

# Import earnings router (simplified)
from api.routers.earnings import router as earnings_router

from pydantic import BaseModel
from typing import List, Dict, Any

logger = logging.getLogger("requiem.api")

# Initialize app
app = FastAPI(title="Requiem API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(earnings_router, prefix="/v1/earnings", tags=["earnings"])

# Global variables
RUNS: dict[str, dict] = {}

@app.get("/")
async def root():
    return {"message": "Requiem Financial Analysis API", "status": "running"}

@app.get("/health")
def health():
    return {"ok": True}

# Basic technical analysis endpoint
@app.post("/api/technical-analysis")
async def technical_analysis(request: Dict[str, Any]):
    """Simplified technical analysis endpoint"""
    try:
        # Basic validation
        if not request.get("query"):
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Use intelligent analyzer
        analyzer = IntelligentAnalyzer()
        result = analyzer.analyze(request["query"])
        
        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Technical analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Basic intelligent analysis endpoint
@app.post("/api/intelligent-analysis")
async def intelligent_analysis(request: Dict[str, Any]):
    """Simplified intelligent analysis endpoint"""
    try:
        if not request.get("query"):
            raise HTTPException(status_code=400, detail="Query is required")
        
        analyzer = IntelligentAnalyzer()
        result = analyzer.analyze(request["query"])
        
        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Intelligent analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
