"""
Simple FastAPI app for Railway testing
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

@app.get("/test")
async def test():
    return {"message": "Test endpoint working!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
