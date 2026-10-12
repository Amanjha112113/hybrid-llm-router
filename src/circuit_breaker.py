"""
Circuit Breaker pattern for API resilience.
Prevents cascading failures when external services are degraded.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states following the standard pattern."""
    CLOSED = "closed"        # Normal operation, requests pass through
    OPEN = "open"            # Failure threshold exceeded, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Open circuit after this many failures
    success_threshold: int = 2          # Close circuit after this many successes in half-open
    timeout_seconds: float = 60.0       # Time to wait before trying half-open
    window_seconds: float = 120.0       # Rolling window for failure counting


class CircuitBreaker:
    """
    Circuit breaker for external API calls.
    
    Protects against cascading failures by failing fast when an external
    service is degraded rather than waiting for timeouts.
    
    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Too many failures, reject all requests immediately
    - HALF_OPEN: Testing recovery, allow limited requests through
    
    Example:
        breaker = CircuitBreaker("fireworks-api")
        
        if not breaker.allow_request():
            raise Exception("Circuit breaker open - service unavailable")
        
        try:
            result = await api_call()
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure()
            raise
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.opened_at: Optional[float] = None
        
        self._lock = Lock()
    
    def allow_request(self) -> bool:
        """
        Check if a request should be allowed through the circuit.
        
        Returns:
            True if request should proceed, False if circuit is open
        """
        with self._lock:
            now = time.time()
            
            if self.state == CircuitState.CLOSED:
                # Normal operation
                return True
            
            elif self.state == CircuitState.OPEN:
                # Check if timeout has elapsed
                if self.opened_at and (now - self.opened_at) >= self.config.timeout_seconds:
                    logger.info(f"Circuit breaker '{self.name}': Transitioning to HALF_OPEN after {self.config.timeout_seconds}s")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self.failure_count = 0
                    return True
                else:
                    # Still in timeout period
                    logger.warning(f"Circuit breaker '{self.name}': OPEN - rejecting request")
                    return False
            
            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited requests to test recovery
                return True
            
            return False
    
    def record_success(self):
        """Record a successful request."""
        with self._lock:
            now = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                logger.debug(f"Circuit breaker '{self.name}': Success in HALF_OPEN ({self.success_count}/{self.config.success_threshold})")
                
                if self.success_count >= self.config.success_threshold:
                    logger.info(f"Circuit breaker '{self.name}': Transitioning to CLOSED after {self.success_count} successes")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.last_failure_time = None
                    self.opened_at = None
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                if self.failure_count > 0:
                    logger.debug(f"Circuit breaker '{self.name}': Resetting failure count after success")
                    self.failure_count = 0
                    self.last_failure_time = None
    
    def record_failure(self):
        """Record a failed request."""
        with self._lock:
            now = time.time()
            self.last_failure_time = now
            
            if self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit breaker '{self.name}': Failure in HALF_OPEN - reopening circuit")
                self.state = CircuitState.OPEN
                self.opened_at = now
                self.failure_count = self.config.failure_threshold
                self.success_count = 0
            
            elif self.state == CircuitState.CLOSED:
                self.failure_count += 1
                logger.debug(f"Circuit breaker '{self.name}': Failure recorded ({self.failure_count}/{self.config.failure_threshold})")
                
                if self.failure_count >= self.config.failure_threshold:
                    logger.error(f"Circuit breaker '{self.name}': Opening circuit after {self.failure_count} failures")
                    self.state = CircuitState.OPEN
                    self.opened_at = now
    
    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "opened_at": self.opened_at,
            }
    
    def reset(self):
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            logger.info(f"Circuit breaker '{self.name}': Manual reset to CLOSED")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.opened_at = None


# Global circuit breakers for external services
_circuit_breakers = {}
_breakers_lock = Lock()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """
    Get or create a circuit breaker for a named service.
    
    Args:
        name: Service name (e.g., "fireworks-api", "ollama")
        config: Optional configuration (uses defaults if not provided)
    
    Returns:
        CircuitBreaker instance
    """
    with _breakers_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def get_all_breakers_status() -> dict:
    """Get status of all circuit breakers."""
    with _breakers_lock:
        return {
            name: breaker.get_status()
            for name, breaker in _circuit_breakers.items()
        }
