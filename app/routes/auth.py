from datetime import datetime, timedelta
from typing import Dict

from fastapi import APIRouter, HTTPException, status
from fastapi import Body
from jose import jwt

from ..config import settings

router = APIRouter()


@router.post("/login")
async def login(username: str = Body(...), password: str = Body(...)) -> Dict:
	if not settings.local_admin_enabled:
		raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Local admin disabled")
	if username != settings.local_admin_username or password != settings.local_admin_password:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
	now = datetime.utcnow()
	claims = {
		"sub": f"local:{settings.local_admin_username}",
		"email": settings.local_admin_email,
		"name": settings.local_admin_username,
		"roles": ["admin"],
		"iss": "local",
		"iat": int(now.timestamp()),
		"exp": int((now + timedelta(hours=8)).timestamp()),
	}
	token = jwt.encode(claims, settings.secret_key, algorithm="HS256")
	return {"access_token": token, "token_type": "bearer"}
