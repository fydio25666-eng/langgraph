"""Minimal API-token authentication for protected HTTP routes."""
import hmac
import os

from dotenv import load_dotenv
from fastapi import HTTPException, Request, status


def require_api_token(request: Request) -> None:
    """Require Authorization: Bearer <API_TOKEN>; fail closed when unconfigured."""
    load_dotenv()
    expected = os.getenv("API_TOKEN", "")
    authorization = request.headers.get("Authorization", "")
    scheme, _, provided = authorization.partition(" ")
    if (
        not expected
        or scheme.lower() != "bearer"
        or not provided
        or not hmac.compare_digest(provided, expected)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或缺少 API Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
