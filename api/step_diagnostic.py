"""
Step-by-step diagnostic to identify exact import failure
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import traceback

app = FastAPI(title="Step Diagnostic API", version="1.0.0")

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
    return {"message": "Step Diagnostic API is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy"}

@app.get("/test-step-imports")
async def test_step_imports():
    """Test imports step by step to find exact failure point"""
    results = {}
    
    # Step 1: Basic imports
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
        results["step1_basic"] = "✅ Success"
    except Exception as e:
        results["step1_basic"] = f"❌ {str(e)}"
        return results
    
    # Step 2: Workers API paths
    try:
        from workers.api_paths import StrategySpec
        results["step2_api_paths"] = "✅ Success"
    except Exception as e:
        results["step2_api_paths"] = f"❌ {str(e)}"
        return results
    
    # Step 3: Workers exceptions
    try:
        from workers.exceptions import ValidationError, DataError, InternalError, ExternalAPIError
        results["step3_exceptions"] = "✅ Success"
    except Exception as e:
        results["step3_exceptions"] = f"❌ {str(e)}"
        return results
    
    # Step 4: Workers prompt parser
    try:
        from workers.engine.prompt_parser import parse_prompt, prompt_to_spec_skeleton, validate_spec_skeleton
        results["step4_prompt_parser"] = "✅ Success"
    except Exception as e:
        results["step4_prompt_parser"] = f"❌ {str(e)}"
        return results
    
    # Step 5: Workers planning
    try:
        from workers.engine.planning import plan_generator, plan_executor
        results["step5_planning"] = "✅ Success"
    except Exception as e:
        results["step5_planning"] = f"❌ {str(e)}"
        return results
    
    # Step 6: Workers intelligent analyzer
    try:
        from workers.engine.intelligent_analyzer import IntelligentAnalyzer
        results["step6_intelligent_analyzer"] = "✅ Success"
    except Exception as e:
        results["step6_intelligent_analyzer"] = f"❌ {str(e)}"
        return results
    
    # Step 7: Workers talib tool executor
    try:
        from workers.engine.talib_tool_executor import TALibToolExecutor
        results["step7_talib_tool_executor"] = "✅ Success"
    except Exception as e:
        results["step7_talib_tool_executor"] = f"❌ {str(e)}"
        return results
    
    # Step 8: API routers earnings
    try:
        from api.routers.earnings import router as earnings_router
        results["step8_earnings_router"] = "✅ Success"
    except Exception as e:
        results["step8_earnings_router"] = f"❌ {str(e)}"
        return results
    
    # Step 9: Pydantic and typing
    try:
        from pydantic import BaseModel
        from typing import List, Dict, Any
        results["step9_pydantic_typing"] = "✅ Success"
    except Exception as e:
        results["step9_pydantic_typing"] = f"❌ {str(e)}"
        return results
    
    # Step 10: Try to create the FastAPI app
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        test_app = FastAPI(title="Test API", version="1.0.0")
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        results["step10_fastapi_app"] = "✅ Success"
    except Exception as e:
        results["step10_fastapi_app"] = f"❌ {str(e)}"
        return results
    
    return results

@app.get("/test-specific-imports")
async def test_specific_imports():
    """Test specific problematic imports"""
    results = {}
    
    # Test matplotlib
    try:
        import matplotlib
        results["matplotlib"] = f"✅ {matplotlib.__version__}"
    except Exception as e:
        results["matplotlib"] = f"❌ {str(e)}"
    
    # Test chromadb
    try:
        import chromadb
        results["chromadb"] = f"✅ {chromadb.__version__}"
    except Exception as e:
        results["chromadb"] = f"❌ {str(e)}"
    
    # Test pandas
    try:
        import pandas as pd
        results["pandas"] = f"✅ {pd.__version__}"
    except Exception as e:
        results["pandas"] = f"❌ {str(e)}"
    
    # Test numpy
    try:
        import numpy as np
        results["numpy"] = f"✅ {np.__version__}"
    except Exception as e:
        results["numpy"] = f"❌ {str(e)}"
    
    return results

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
