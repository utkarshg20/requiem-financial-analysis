"""
Minimal main app that gradually adds functionality
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

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
async def root():
    return {"message": "Requiem API is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy"}

# Test basic imports one by one
@app.get("/test-basic-imports")
async def test_basic_imports():
    """Test basic imports without complex dependencies"""
    results = {}
    
    try:
        import pandas as pd
        results["pandas"] = f"✅ {pd.__version__}"
    except Exception as e:
        results["pandas"] = f"❌ {str(e)}"
    
    try:
        import numpy as np
        results["numpy"] = f"✅ {np.__version__}"
    except Exception as e:
        results["numpy"] = f"❌ {str(e)}"
    
    try:
        import openai
        results["openai"] = f"✅ {openai.__version__}"
    except Exception as e:
        results["openai"] = f"❌ {str(e)}"
    
    return results

# Test ChromaDB specifically
@app.get("/test-chromadb")
async def test_chromadb():
    """Test ChromaDB import specifically"""
    try:
        import chromadb
        return {"chromadb": f"✅ {chromadb.__version__}", "status": "success"}
    except Exception as e:
        return {"chromadb": f"❌ {str(e)}", "status": "failed"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
