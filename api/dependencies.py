"""
FastAPI dependencies for the NSE Stock Prediction API.

This module provides reusable dependencies for:
- Authentication and authorization
- Rate limiting and throttling
- Caching and cache management
- Database connections
- Request validation and preprocessing
- Response formatting and compression
- Logging and monitoring
- Error handling and recovery

"""

import asyncio
import hashlib
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional

import redis
from fastapi import BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

from config import settings
from modules.prediction_model import StockPredictionModel
from modules.stock_analyzer import StockAnalyzer
from modules.stock_data_fetcher import StockDataFetcher
from modules.technical_indicators import TechnicalIndicators

# Security dependencies
security = HTTPBearer(auto_error=False)


class RateLimiter:
    """
    Token bucket rate limiter for API endpoints.
    """

    def __init__(self):
        self.clients = defaultdict(
            lambda: {"tokens": settings.rate_limit_requests, "last_update": time.time()}
        )
        self.max_tokens = settings.rate_limit_requests
        self.refill_rate = settings.rate_limit_requests / settings.rate_limit_period

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.
        """
        now = time.time()
        client = self.clients[client_id]

        # Add tokens based on time passed
        time_passed = now - client["last_update"]
        client["tokens"] = min(self.max_tokens, client["tokens"] + time_passed * self.refill_rate)
        client["last_update"] = now

        # Check if request is allowed
        if client["tokens"] >= 1:
            client["tokens"] -= 1
            return True
        return False

    def get_retry_after(self, client_id: str) -> int:
        """
        Get retry after time in seconds.
        """
        client = self.clients[client_id]
        tokens_needed = 1 - client["tokens"]
        return max(1, int(tokens_needed / self.refill_rate))


# Global rate limiter instance
rate_limiter = RateLimiter()


class Cache:
    """
    Simple in-memory cache with TTL support.
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._expiry: Dict[str, datetime] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        """
        if key in self._cache:
            if key in self._expiry and datetime.now() > self._expiry[key]:
                # Expired
                del self._cache[key]
                del self._expiry[key]
                return None
            return self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """
        Set value in cache with TTL.
        """
        self._cache[key] = value
        self._expiry[key] = datetime.now() + timedelta(seconds=ttl)

    def delete(self, key: str) -> None:
        """
        Delete key from cache.
        """
        self._cache.pop(key, None)
        self._expiry.pop(key, None)

    def clear(self) -> None:
        """
        Clear entire cache.
        """
        self._cache.clear()
        self._expiry.clear()

    def size(self) -> int:
        """
        Get cache size.
        """
        # Clean expired items first
        now = datetime.now()
        expired_keys = [k for k, exp in self._expiry.items() if now > exp]
        for key in expired_keys:
            self.delete(key)
        return len(self._cache)


# Global cache instance
cache = Cache()


# Redis cache (optional)
redis_client = None
try:
    if hasattr(settings, "redis_url") and settings.redis_url:
        redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        # Test connection
        redis_client.ping()
        logger.info("Redis cache connected successfully")
except Exception as e:
    logger.warning(f"Redis not available, using in-memory cache: {e}")
    redis_client = None


class AnalyzerPool:
    """
    Pool of analyzer instances for better performance.
    """

    def __init__(self, pool_size: int = 5):
        self.pool = asyncio.Queue(maxsize=pool_size)
        self.pool_size = pool_size
        self._initialize_pool()

    def _initialize_pool(self):
        """
        Initialize the analyzer pool.
        """
        for _ in range(self.pool_size):
            analyzer = StockAnalyzer()
            try:
                self.pool.put_nowait(analyzer)
            except asyncio.QueueFull:
                break

    async def get_analyzer(self) -> StockAnalyzer:
        """
        Get analyzer from pool or create new one.
        """
        try:
            return await asyncio.wait_for(self.pool.get(), timeout=1.0)
        except asyncio.TimeoutError:
            # Pool is empty, create new analyzer
            return StockAnalyzer()

    async def return_analyzer(self, analyzer: StockAnalyzer) -> None:
        """
        Return analyzer to pool.
        """
        try:
            self.pool.put_nowait(analyzer)
        except asyncio.QueueFull:
            # Pool is full, analyzer will be garbage collected
            pass


# Global analyzer pool
analyzer_pool = AnalyzerPool()


# Dependency functions
async def get_client_id(request: Request) -> str:
    """
    Get client identifier for rate limiting.
    """
    # Use IP address as client ID (in production, use API key or user ID)
    client_ip = request.client.host
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    return client_ip


async def check_rate_limit(client_id: str = Depends(get_client_id)) -> None:
    """
    Check rate limit for client.
    """
    if not rate_limiter.is_allowed(client_id):
        retry_after = rate_limiter.get_retry_after(client_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "retry_after": retry_after,
                "limit": settings.rate_limit_requests,
                "period": settings.rate_limit_period,
            },
            headers={"Retry-After": str(retry_after)},
        )


async def validate_stock_symbol(symbol: str) -> str:
    """
    Validate and normalize stock symbol.
    """
    if not symbol or len(symbol.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Stock symbol cannot be empty",
        )

    # Normalize symbol
    symbol = symbol.strip().upper()
