import sys
import os
from contextlib import asynccontextmanager

# Add parent directory to path to import existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import router as api_router
from backend.camera_manager import camera_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    try:
        camera_manager.initialize()
    except Exception as e:
        print(f"Error initializing camera: {e}")

    yield

    # Shutdown
    print("Shutting down...")
    camera_manager.release()

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Router
app.include_router(api_router, prefix="/api")

# Serve frontend static files (will be populated after build)
# app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
