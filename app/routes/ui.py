from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
	return templates.TemplateResponse("login.html", {"request": request})


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
	return templates.TemplateResponse("dashboard.html", {"request": request})
