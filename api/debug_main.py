"""
Debug FastAPI app for Railway troubleshooting
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

app = FastAPI(title="Requiem Debug API", version="1.0.0")

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
    return {"message": "Requiem Debug API is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy"}

@app.get("/debug")
async def debug():
    """Debug endpoint to check what's working"""
    debug_info = {
        "python_version": sys.version,
        "environment_variables": {
            "PORT": os.environ.get("PORT"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
        "working_directory": os.getcwd(),
        "files_in_current_dir": os.listdir("."),
        "files_in_api_dir": os.listdir("api") if os.path.exists("api") else "api directory not found",
    }
    return debug_info

@app.get("/test-imports")
async def test_imports():
    """Test critical imports one by one"""
    results = {}
    
    try:
        import fastapi
        results["fastapi"] = f"✅ {fastapi.__version__}"
    except Exception as e:
        results["fastapi"] = f"❌ {str(e)}"
    
    try:
        import uvicorn
        results["uvicorn"] = f"✅ {uvicorn.__version__}"
    except Exception as e:
        results["uvicorn"] = f"❌ {str(e)}"
    
    try:
        import pandas
        results["pandas"] = f"✅ {pandas.__version__}"
    except Exception as e:
        results["pandas"] = f"❌ {str(e)}"
    
    try:
        import numpy
        results["numpy"] = f"✅ {numpy.__version__}"
    except Exception as e:
        results["numpy"] = f"❌ {str(e)}"
    
    try:
        import openai
        results["openai"] = f"✅ {openai.__version__}"
    except Exception as e:
        results["openai"] = f"❌ {str(e)}"
    
    try:
        import chromadb
        results["chromadb"] = f"✅ {chromadb.__version__}"
    except Exception as e:
        results["chromadb"] = f"❌ {str(e)}"
    
    return results

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
