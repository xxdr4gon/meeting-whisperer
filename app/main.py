import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from prometheus_fastapi_instrumentator import Instrumentator

from .config import settings
from .routes import api
from .routes import auth as auth_routes
from .routes import ui as ui_routes

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Transcription Service", version="1.0.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=settings.allowed_origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(auth_routes.router, prefix="/api/auth", tags=["auth"]) 
app.include_router(api.router, prefix="/api", tags=["api"]) 
app.include_router(ui_routes.router, tags=["ui"]) 

@app.get("/health")
async def health():
	return {"status": "ok"}

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

if STATIC_DIR.exists():
	app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
	return RedirectResponse(url="/login")
