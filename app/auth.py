import json
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt

from .config import settings


class JWKSCache:
	def __init__(self, ttl_seconds: int) -> None:
		self.ttl_seconds = ttl_seconds
		self.last_fetch_ts: float = 0.0
		self.jwks: Dict[str, Any] | None = None

	async def get_jwks(self, jwks_uri: str) -> Dict[str, Any]:
		now = time.time()
		if self.jwks and (now - self.last_fetch_ts) < self.ttl_seconds:
			return self.jwks
		async with httpx.AsyncClient(timeout=10.0) as client:
			resp = await client.get(jwks_uri)
			resp.raise_for_status()
			self.jwks = resp.json()
			self.last_fetch_ts = now
			return self.jwks


bearer_scheme = HTTPBearer(auto_error=True)
_jwks_cache = JWKSCache(ttl_seconds=settings.oidc_jwks_cache_ttl)


async def _get_openid_config() -> Dict[str, Any]:
    # If OIDC is not configured, return empty config instead of 503
    if not settings.oidc_discovery_url:
        return {}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(settings.oidc_discovery_url)
        resp.raise_for_status()
        return resp.json()


def _find_key(jwks: Dict[str, Any], kid: str) -> Optional[Dict[str, Any]]:
	for k in jwks.get("keys", []):
		if k.get("kid") == kid:
			return k
	return None


def _json_path_get(payload: Dict[str, Any], path: str) -> Any:
	parts = [p for p in path.split(".") if p]
	cur: Any = payload
	for p in parts:
		if isinstance(cur, dict) and p in cur:
			cur = cur[p]
		else:
			return None
	return cur


def _map_roles(external_roles: Any) -> List[str]:
	try:
		mapping = json.loads(settings.__dict__.get("role_mapping_json", settings.__dict__.get("ROLE_MAPPING_JSON", "{}")))
	except Exception:
		mapping = {}
	mapped: List[str] = []
	if isinstance(external_roles, list):
		for internal, values in mapping.items():
			if any(v in external_roles for v in values):
				mapped.append(internal)
	return list(sorted(set(mapped)))


async def _try_decode_local(token: str) -> Optional[Dict[str, Any]]:
	try:
		claims = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
		if claims.get("iss") == "local":
			# trust roles claim if present, else default admin when local
			roles = claims.get("roles") or ["admin"]
			return {
				"sub": claims.get("sub"),
				"email": claims.get("email"),
				"name": claims.get("name"),
				"roles": roles,
			}
	except Exception:
		return None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> Dict[str, Any]:
	token = credentials.credentials
	local_user = await _try_decode_local(token)
	if settings.local_admin_enabled and local_user:
		return local_user

	cfg = await _get_openid_config()
	jwks_uri = cfg.get("jwks_uri")
	if not jwks_uri:
		# No IdP configured; reject with 401 instead of 503 to avoid false service errors
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No IdP configured and local token not provided")
	unverified = jwt.get_unverified_header(token)
	kid = unverified.get("kid")
	if not kid:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing kid")
	jwks = await _jwks_cache.get_jwks(jwks_uri)
	key = _find_key(jwks, kid)
	if not key:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown key")

	issuer = settings.oidc_issuer
	audience = settings.oidc_audience or settings.oidc_client_id
	algorithms = [a.strip() for a in settings.oidc_allowed_algs.split(",")]
	try:
		claims = jwt.decode(token, key, algorithms=algorithms, audience=audience, issuer=issuer)
	except Exception as exc:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {exc}")

	role_claim_path = getattr(settings, "role_claim_path", None) or settings.__dict__.get("ROLE_CLAIM_PATH", "realm_access.roles")
	external_roles = _json_path_get(claims, role_claim_path) or []
	roles = _map_roles(external_roles)
	user = {
		"sub": claims.get("sub"),
		"email": claims.get("email"),
		"name": claims.get("name") or claims.get("preferred_username"),
		"roles": roles,
	}
	return user


def require_role(required: str):
	async def _dep(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
		if required not in user.get("roles", []):
			raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
		return user
	return _dep
