"""
src/auth_middleware.py

Simple API key authentication and rate limiting middleware.

P0 Security Fix: Add authentication to prevent unauthorized access
and rate limiting to prevent abuse of cloud API resources.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────

# API keys from environment (comma-separated for multiple keys)
# Format: API_KEY_1,API_KEY_2,API_KEY_3
VALID_API_KEYS = set(
    k.strip()
    for k in os.environ.get("VALID_API_KEYS", "").split(",")
    if k.strip()
)

# Rate limiting: requests per minute per API key
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))

# Allow anonymous access (set to "false" to require auth)
ALLOW_ANONYMOUS = os.environ.get("ALLOW_ANONYMOUS", "true").lower() == "true"


# ── Rate Limiting State ────────────────────────────────────────────────────

class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)
    
    def check_rate_limit(self, key: str) -> tuple[bool, int]:
        """
        Check if key is within rate limit.
        
        Returns:
            (allowed, remaining_requests)
        """
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Remove expired requests
        self.requests[key] = [
            ts for ts in self.requests[key]
            if ts > window_start
        ]
        
        # Check limit
        current_count = len(self.requests[key])
        allowed = current_count < self.requests_per_minute
        
        if allowed:
            self.requests[key].append(now)
        
        remaining = max(0, self.requests_per_minute - current_count - 1)
        return allowed, remaining


rate_limiter = RateLimiter(requests_per_minute=RATE_LIMIT_PER_MINUTE)


# ── Authentication ─────────────────────────────────────────────────────────

security = HTTPBearer(auto_error=False)


def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials]) -> str:
    """
    Verify API key from Authorization header.
    
    Returns:
        api_key (for logging/rate limiting)
    
    Raises:
        HTTPException: If authentication fails
    """
    # If anonymous access is allowed and no key provided
    if ALLOW_ANONYMOUS and not credentials:
        return "anonymous"
    
    # If keys are configured, require authentication
    if VALID_API_KEYS:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Include 'Authorization: Bearer <key>' header.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        api_key = credentials.credentials
        
        if api_key not in VALID_API_KEYS:
            logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        logger.debug(f"Authenticated with API key: {api_key[:8]}...")
        return api_key
    
    # No keys configured - allow access but warn
    if not ALLOW_ANONYMOUS:
        logger.warning(
            "ALLOW_ANONYMOUS=false but no VALID_API_KEYS configured. "
            "Allowing access anyway — set VALID_API_KEYS in .env to secure this API."
        )
    
    return credentials.credentials if credentials else "anonymous"


# ── Middleware ─────────────────────────────────────────────────────────────

class AuthRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Combined authentication and rate limiting middleware.
    
    Applies to all requests except:
    - /docs (OpenAPI docs)
    - /redoc
    - /openapi.json
    - /api/health (health check)
    """
    
    EXCLUDED_PATHS = {"/docs", "/redoc", "/openapi.json", "/api/health"}
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth/rate limit for excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)
        
        # ── Authentication ─────────────────────────────────────────────────
        try:
            # Extract API key from Authorization header
            auth_header = request.headers.get("Authorization")
            credentials = None
            
            if auth_header and auth_header.startswith("Bearer "):
                from fastapi.security import HTTPAuthorizationCredentials
                credentials = HTTPAuthorizationCredentials(
                    scheme="Bearer",
                    credentials=auth_header[7:]  # Strip "Bearer "
                )
            
            api_key = verify_api_key(credentials)
            
            # Store API key in request state for logging
            request.state.api_key = api_key
            
        except HTTPException as e:
            # Return auth error immediately
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail},
                headers=e.headers or {},
            )
        
        # ── Rate Limiting ──────────────────────────────────────────────────
        allowed, remaining = rate_limiter.check_rate_limit(api_key)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for key: {api_key[:8]}...")
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded: {RATE_LIMIT_PER_MINUTE} requests per minute",
                    "retry_after": 60,
                },
                headers={
                    "X-RateLimit-Limit": str(RATE_LIMIT_PER_MINUTE),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": "60",
                },
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_PER_MINUTE)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response


# ── FastAPI Dependency (alternative to middleware) ─────────────────────────

async def get_api_key(credentials: Optional[HTTPAuthorizationCredentials] = security) -> str:
    """
    FastAPI dependency for route-level authentication.
    
    Usage:
        @app.post("/api/protected")
        async def protected_route(api_key: str = Depends(get_api_key)):
            # Only authenticated users reach here
            ...
    """
    return verify_api_key(credentials)
