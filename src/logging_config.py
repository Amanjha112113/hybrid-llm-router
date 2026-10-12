"""
src/logging_config.py

Structured JSON logging configuration for production observability.

P1 Fix: Replace plain text logging with structured JSON for log aggregation
(Datadog, CloudWatch, ELK, etc.)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


def configure_structured_logging(log_level: str = "INFO", json_logs: bool = True):
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: If True, output JSON. If False, use pretty console format.
    """
    
    if not STRUCTLOG_AVAILABLE:
        # Fallback to standard logging with improved format
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            stream=sys.stdout,
        )
        logging.warning(
            "structlog not installed. Using standard logging. "
            "Install with: pip install structlog"
        )
        return
    
    # Configure processors based on output format
    if json_logs:
        processors = [
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging to output through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )


def get_logger(name: str) -> Any:
    """
    Get a logger instance.
    
    Returns structlog logger if available, otherwise standard logger.
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)
