"""
FastAPI main application for NSE Stock Prediction API.

This module creates and configures the FastAPI application with:
- API routing and middleware
- CORS and security headers
- Request/response logging
- Error handling
- OpenAPI documentation
- Health monitoring
- Graceful shutdown

"""

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from config import settings

from . import (
    API_CONTACT,
    API_DESCRIPTION,
    API_LICENSE,
    API_TAGS_METADATA,
    API_TITLE,
    API_VERSION,
)
from .dependencies import cleanup_dependencies, performance_monitor, request_logger
from .routes import router


# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.
    """
    # Startup
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    # Initialize components
    startup_time = time.time()
    logger.info(f"start time {startup_time}")

    try:
        # Add any startup tasks here
        logger.info("Application startup completed")
        yield
    finally:
        # Shutdown
        logger.info("Shutting down application...")
        await cleanup_dependencies()
        logger.info("Application shutdown completed")


# Create FastAPI application
def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    """

    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        contact=API_CONTACT,
        license_info=API_LICENSE,
        openapi_tags=API_TAGS_METADATA,
        docs_url=settings.api_docs_url,
        redoc_url=settings.api_redoc_url,
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # Configure middleware
    setup_middleware(app)

    # Configure routes
    setup_routes(app)

    # Configure exception handlers
    setup_exception_handlers(app)

    # Configure OpenAPI
    setup_openapi(app)

    return app


def setup_middleware(app: FastAPI) -> None:
    """
    Configure application middleware.
    """

    # Trusted hosts (security)
    if not settings.debug:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins
        if settings.environment == "development"
        else ["https://your-domain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"],
    )

    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Custom middleware for logging and monitoring
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        """
        Log requests and responses with timing.
        """
        start_time = time.time()
        request_id = f"req_{int(start_time * 1000000)}"

        # Add request ID to headers
        request.state.request_id = request_id

        # Process request
        try:
            response = await call_next(request)

            # Calculate response time
            process_time = time.time() - start_time

            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{process_time:.3f}s"

            # Log request
            logger.info(
                f"{request.method} {request.url.path} - "
                f"{response.status_code} - {process_time:.3f}s - "
                f"ID: {request_id}"
            )

            # Record metrics
            endpoint = f"{request.method} {request.url.path}"
            performance_monitor.record_request(
                endpoint, process_time, 200 <= response.status_code < 400
            )

            return response

        except Exception as e:
            process_time = time.time() - start_time

            logger.error(
                f"{request.method} {request.url.path} - "
                f"ERROR - {process_time:.3f}s - "
                f"ID: {request_id} - Error: {str(e)}"
            )

            # Record error metrics
            endpoint = f"{request.method} {request.url.path}"
            performance_monitor.record_request(endpoint, process_time, False)

            raise e

    # Security headers middleware
    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        """
        Add security headers to all responses.
        """
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        if not settings.debug:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


def setup_routes(app: FastAPI) -> None:
    """
    Configure application routes.
    """

    # Include API routes
    app.include_router(
        router,
        prefix="/api/v1",
        responses={
            404: {"description": "Not found"},
            422: {"description": "Validation error"},
            429: {"description": "Rate limit exceeded"},
            500: {"description": "Internal server error"},
        },
    )

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint - redirect to docs."""
        return RedirectResponse(url=settings.api_docs_url)

    # Health check endpoint (outside versioning)
    @app.get("/health", include_in_schema=False)
    async def health():
        """
        Simple health check endpoint.
        """
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": API_VERSION,
            "environment": settings.environment,
        }

    # API info endpoint
    @app.get("/api/info", include_in_schema=False)
    async def api_info():
        """
        Get API information.
        """
        return {
            "name": API_TITLE,
            "version": API_VERSION,
            "description": "NSE Stock Prediction API",
            "environment": settings.environment,
            "docs_url": settings.api_docs_url,
            "redoc_url": settings.api_redoc_url,
            "openapi_url": "/api/v1/openapi.json",
            "contact": API_CONTACT,
            "license": API_LICENSE,
        }

    # Metrics endpoint (for monitoring)
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        """
        Get application metrics (Prometheus format in production).
        """
        metrics = performance_monitor.get_metrics()

        # In production, return Prometheus format
        if settings.environment == "production":
            # Convert to Prometheus format
            prometheus_metrics = []
            prometheus_metrics.append("# HELP app_uptime_seconds Application uptime in seconds")
            prometheus_metrics.append("# TYPE app_uptime_seconds gauge")
            prometheus_metrics.append(f"app_uptime_seconds {metrics.get('uptime', 0)}")

            prometheus_metrics.append("# HELP app_requests_total Total number of requests")
            prometheus_metrics.append("# TYPE app_requests_total counter")
            prometheus_metrics.append(f"app_requests_total {metrics.get('total_requests', 0)}")

            prometheus_metrics.append("# HELP app_errors_total Total number of errors")
            prometheus_metrics.append("# TYPE app_errors_total counter")
            prometheus_metrics.append(f"app_errors_total {metrics.get('total_errors', 0)}")

            return Response(content="\n".join(prometheus_metrics), media_type="text/plain")

        # Development format
        return metrics

    # Static files (if needed)
    try:
        from pathlib import Path

        static_dir = Path("static")
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory="static"), name="static")
    except Exception:
        pass  # Static files not available


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Configure custom exception handlers.
    """

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """
        Handle HTTP exceptions with consistent format.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "error_code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(422)
    async def validation_exception_handler(request: Request, exc):
        """
        Handle validation errors.
        """
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "error_code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors() if hasattr(exc, "errors") else str(exc),
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(500)
    async def internal_server_error_handler(request: Request, exc: Exception):
        """
        Handle internal server errors.
        """
        logger.error(f"Internal server error: {str(exc)}")

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred",
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(429)
    async def rate_limit_handler(request: Request, exc: HTTPException):
        """
        Handle rate limit errors.
        """
        return JSONResponse(
            status_code=429,
            content={
                "status": "error",
                "error_code": "RATE_LIMIT_EXCEEDED",
                "message": "Rate limit exceeded. Please try again later.",
                "retry_after": exc.headers.get("Retry-After") if exc.headers else None,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path),
                "request_id": getattr(request.state, "request_id", None),
            },
            headers=exc.headers if exc.headers else {},
        )


def setup_openapi(app: FastAPI) -> None:
    """
    Configure OpenAPI schema with custom settings.
    """

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            contact=app.contact,
            license_info=app.license_info,
            tags=app.openapi_tags,
        )

        # Add custom schema properties
        openapi_schema["info"]["x-logo"] = {
            "url": "https://your-domain.com/logo.png",
            "altText": "NSE Stock Prediction API",
        }

        # Add server information
        openapi_schema["servers"] = [
            {
                "url": f"http://localhost:{settings.api_port}",
                "description": "Development server",
            },
            {"url": "https://api.your-domain.com", "description": "Production server"},
        ]

        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "Enter your API token",
            },
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "Enter your API key",
            },
        }

        # Add global security requirement (optional)
        # openapi_schema["security"] = [{"BearerAuth": []}]

        # Add custom examples
        openapi_schema["components"]["examples"] = {
            "TCS_Analysis": {
                "summary": "TCS Stock Analysis",
                "description": "Example analysis for TCS stock",
                "value": {
                    "symbol": "TCS",
                    "current_price": 3456.78,
                    "predicted_price": 3500.25,
                    "technical_signal": "Bullish",
                },
            },
            "Multiple_Stocks": {
                "summary": "Multiple Stock Comparison",
                "description": "Example comparison of multiple stocks",
                "value": {
                    "symbols": ["TCS", "INFY", "WIPRO"],
                    "comparison_type": "prediction",
                },
            },
        }

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi


# Create the application instance
app = create_app()


# Development server runner
def run_dev_server():
    """
    Run development server with auto-reload.
    """
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
        access_log=True,
        reload_dirs=["api", "modules"],
        reload_includes=["*.py"],
    )


# Production server configuration
def get_uvicorn_config() -> Dict[str, Any]:
    """
    Get Uvicorn configuration for production.
    """
    return {
        "app": "api.main:app",
        "host": settings.api_host,
        "port": settings.api_port,
        "workers": settings.api_workers,
        "log_level": settings.log_level.lower(),
        "access_log": True,
        "keep_alive": 30,
        "timeout_keep_alive": 30,
        "timeout_graceful_shutdown": 10,
        "limit_concurrency": 1000,
        "limit_max_requests": 10000,
        "backlog": 2048,
    }


# Application startup event (alternative to lifespan for older FastAPI versions)
@app.on_event("startup")
async def startup_event():
    """
    Application startup tasks.
    """
    logger.info("Application startup event triggered")

    # Log configuration
    logger.info(f"API Host: {settings.api_host}:{settings.api_port}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Docs URL: {settings.api_docs_url}")
    logger.info(f"Cache TTL: {settings.cache_ttl}s")
    logger.info(
        f"Rate Limit: {settings.rate_limit_requests} requests per {settings.rate_limit_period}s"
    )

    # Initialize any required services here
    logger.info("Services initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown tasks.
    """
    logger.info("Application shutdown event triggered")
    await cleanup_dependencies()


# CLI command for running the server
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        # Development mode
        logger.info("Starting development server...")
        run_dev_server()
    else:
        # Production mode
        logger.info("Starting production server...")
        config = get_uvicorn_config()
        uvicorn.run(**config)


# Export the app instance for external use
__all__ = ["app", "create_app", "run_dev_server", "get_uvicorn_config"]
