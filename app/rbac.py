from typing import Dict


def user_has_role(user: Dict, role: str) -> bool:
	return role in (user.get("roles") or [])
