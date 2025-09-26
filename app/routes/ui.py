from pathlib import Path
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
	return templates.TemplateResponse("login.html", {"request": request})


from ..auth import get_current_user


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    # Check if user is authenticated by looking for token in cookies or redirect to login
    # The frontend will handle the actual authentication via localStorage
    return templates.TemplateResponse("dashboard.html", {"request": request})
