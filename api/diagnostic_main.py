"""
Diagnostic app to identify health check issues
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import traceback

app = FastAPI(title="Diagnostic API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Diagnostic API is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy"}

@app.get("/test-imports")
async def test_imports():
    """Test all imports from main.py"""
    results = {}
    
    # Test basic imports
    try:
        from fastapi import FastAPI, HTTPException, UploadFile, File
        from fastapi.middleware.cors import CORSMiddleware
        from datetime import datetime
        import uuid
        import logging
        import os
        import zipfile
        import tempfile
        import numpy as np
        from dotenv import load_dotenv
        results["basic_imports"] = "✅ Success"
    except Exception as e:
        results["basic_imports"] = f"❌ {str(e)}"
    
    # Test workers imports
    try:
        from workers.api_paths import StrategySpec
        results["workers_api_paths"] = "✅ Success"
    except Exception as e:
        results["workers_api_paths"] = f"❌ {str(e)}"
    
    try:
        from workers.exceptions import ValidationError, DataError, InternalError, ExternalAPIError
        results["workers_exceptions"] = "✅ Success"
    except Exception as e:
        results["workers_exceptions"] = f"❌ {str(e)}"
    
    try:
        from workers.engine.prompt_parser import parse_prompt, prompt_to_spec_skeleton, validate_spec_skeleton
        results["workers_prompt_parser"] = "✅ Success"
    except Exception as e:
        results["workers_prompt_parser"] = f"❌ {str(e)}"
    
    try:
        from workers.engine.planning import plan_generator, plan_executor
        results["workers_planning"] = "✅ Success"
    except Exception as e:
        results["workers_planning"] = f"❌ {str(e)}"
    
    try:
        from workers.engine.intelligent_analyzer import IntelligentAnalyzer
        results["workers_intelligent_analyzer"] = "✅ Success"
    except Exception as e:
        results["workers_intelligent_analyzer"] = f"❌ {str(e)}"
    
    try:
        from workers.engine.talib_tool_executor import TALibToolExecutor
        results["workers_talib_tool_executor"] = "✅ Success"
    except Exception as e:
        results["workers_talib_tool_executor"] = f"❌ {str(e)}"
    
    try:
        from api.routers.earnings import router as earnings_router
        results["api_routers_earnings"] = "✅ Success"
    except Exception as e:
        results["api_routers_earnings"] = f"❌ {str(e)}"
    
    try:
        from pydantic import BaseModel
        from typing import List, Dict, Any
        results["pydantic_typing"] = "✅ Success"
    except Exception as e:
        results["pydantic_typing"] = f"❌ {str(e)}"
    
    return results

@app.get("/test-main-import")
async def test_main_import():
    """Test importing the main module directly"""
    try:
        import api.main
        return {"status": "success", "message": "Main module imported successfully"}
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
