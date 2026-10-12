"""
src/request_id_middleware.py

Request ID tracing middleware for distributed request tracking.

P2 Fix: Add request IDs to trace requests through the system
"""

from __future__ import annotations

import uuid
from starlette.middleware.base import BaseHTTPMiddleware

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request IDs to all requests for tracing.
    
    - Accepts X-Request-ID header from client
    - Generates UUID if not provided
    - Adds to response headers
    - Binds to structlog context for automatic inclusion in all logs
    """
    
    async def dispatch(self, request, call_next):
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Store in request state for access in route handlers
        request.state.request_id = request_id
        
        # Bind to structlog context if available
        if STRUCTLOG_AVAILABLE:
            structlog.contextvars.bind_contextvars(request_id=request_id)
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers for client tracking
        response.headers["X-Request-ID"] = request_id
        
        # Clear structlog context
        if STRUCTLOG_AVAILABLE:
            structlog.contextvars.clear_contextvars()
        
        return response
