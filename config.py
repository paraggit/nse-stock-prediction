"""
Configuration management using Pydantic Settings.
"""

import os
from pathlib import Path
from typing import List, Optional

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    try:
        from pydantic import BaseSettings, Field
    except ImportError:
        # Simple fallback if pydantic is not available
        print("Warning: Pydantic not available, using simple configuration")

        class SimpleSettings:
            def __init__(self):
                # Application
                self.app_name = os.getenv("APP_NAME", "NSE Stock Prediction API")
                self.app_version = os.getenv("APP_VERSION", "1.0.0")
                self.environment = os.getenv("ENVIRONMENT", "development")
                self.debug = os.getenv("DEBUG", "false").lower() == "true"

                # API Configuration
                self.api_host = os.getenv("API_HOST", "0.0.0.0")
                self.api_port = int(os.getenv("API_PORT", "8000"))
                self.api_workers = int(os.getenv("API_WORKERS", "1"))
                self.api_reload = os.getenv("API_RELOAD", "true").lower() == "true"
                self.api_docs_url = os.getenv("API_DOCS_URL", "/docs")
                self.api_redoc_url = os.getenv("API_REDOC_URL", "/redoc")

                # Security
                self.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
                self.allowed_hosts = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
                self.cors_origins = os.getenv(
                    "CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
                ).split(",")

                # Database (Optional)
                self.database_url = os.getenv("DATABASE_URL", "sqlite:///./stock_data.db")

                # Redis (Optional)
                self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                self.cache_ttl = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes

                # Model Configuration
                self.default_model_type = os.getenv("DEFAULT_MODEL_TYPE", "random_forest")
                self.model_save_path = Path(os.getenv("MODEL_SAVE_PATH", "./models/saved_models"))
                self.prediction_cache_ttl = int(os.getenv("PREDICTION_CACHE_TTL", "60"))
                self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
                self.max_workers = int(os.getenv("MAX_WORKERS", "4"))

                # Data Configuration
                self.data_cache_path = Path(os.getenv("DATA_CACHE_PATH", "./data/cache"))
                self.data_fetch_timeout = int(os.getenv("DATA_FETCH_TIMEOUT", "30"))
                self.yahoo_finance_delay = float(os.getenv("YAHOO_FINANCE_DELAY", "1.0"))
                self.max_retries = int(os.getenv("MAX_RETRIES", "3"))

                # External APIs (Optional)
                self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
                self.polygon_api_key = os.getenv("POLYGON_API_KEY")
                self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")

                # Logging
                self.log_level = os.getenv("LOG_LEVEL", "INFO")
                self.log_file_path = Path(os.getenv("LOG_FILE_PATH", "./logs/app.log"))
                self.log_max_size = os.getenv("LOG_MAX_SIZE", "10MB")
                self.log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

                # Rate Limiting
                self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
                self.rate_limit_period = int(os.getenv("RATE_LIMIT_PERIOD", "60"))  # seconds

                # ML Model Settings
                self.feature_importance_threshold = float(
                    os.getenv("FEATURE_IMPORTANCE_THRESHOLD", "0.01")
                )
                self.cross_validation_folds = int(os.getenv("CROSS_VALIDATION_FOLDS", "5"))
                self.hyperparameter_tuning = (
                    os.getenv("HYPERPARAMETER_TUNING", "false").lower() == "true"
                )
                self.auto_retrain = os.getenv("AUTO_RETRAIN", "false").lower() == "true"
                self.retrain_interval = os.getenv("RETRAIN_INTERVAL", "24h")

                # Stock Market Settings
                self.market_timezone = os.getenv("MARKET_TIMEZONE", "Asia/Kolkata")
                self.trading_hours_start = os.getenv("TRADING_HOURS_START", "09:15")
                self.trading_hours_end = os.getenv("TRADING_HOURS_END", "15:30")
                self.weekend_trading = os.getenv("WEEKEND_TRADING", "false").lower() == "true"

                # Create directories
                self.model_save_path.mkdir(parents=True, exist_ok=True)
                self.data_cache_path.mkdir(parents=True, exist_ok=True)
                self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Use simple settings as fallback
        settings = SimpleSettings()

# Try to use Pydantic if available
try:

    class Settings(BaseSettings):
        """
        Application settings with environment variable support.
        """

        # Application
        app_name: str = Field(default="NSE Stock Prediction API", env="APP_NAME")
        app_version: str = Field(default="1.0.0", env="APP_VERSION")
        environment: str = Field(default="development", env="ENVIRONMENT")
        debug: bool = Field(default=False, env="DEBUG")

        # API Configuration
        api_host: str = Field(default="0.0.0.0", env="API_HOST")
        api_port: int = Field(default=8000, env="API_PORT")
        api_workers: int = Field(default=1, env="API_WORKERS")
        api_reload: bool = Field(default=True, env="API_RELOAD")
        api_docs_url: str = Field(default="/docs", env="API_DOCS_URL")
        api_redoc_url: str = Field(default="/redoc", env="API_REDOC_URL")

        # Security
        secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
        allowed_hosts: List[str] = Field(default=["localhost", "127.0.0.1"], env="ALLOWED_HOSTS")
        cors_origins: List[str] = Field(
            default=["http://localhost:3000", "http://localhost:8080"],
            env="CORS_ORIGINS",
        )

        # Database (Optional)
        database_url: str = Field(default="sqlite:///./stock_data.db", env="DATABASE_URL")

        # Redis (Optional)
        redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
        cache_ttl: int = Field(default=300, env="CACHE_TTL")  # 5 minutes

        # Model Configuration
        default_model_type: str = Field(default="random_forest", env="DEFAULT_MODEL_TYPE")
        model_save_path: Path = Field(default=Path("./models/saved_models"), env="MODEL_SAVE_PATH")
        prediction_cache_ttl: int = Field(default=60, env="PREDICTION_CACHE_TTL")
        batch_size: int = Field(default=32, env="BATCH_SIZE")
        max_workers: int = Field(default=4, env="MAX_WORKERS")

        # Data Configuration
        data_cache_path: Path = Field(default=Path("./data/cache"), env="DATA_CACHE_PATH")
        data_fetch_timeout: int = Field(default=30, env="DATA_FETCH_TIMEOUT")
        yahoo_finance_delay: float = Field(default=1.0, env="YAHOO_FINANCE_DELAY")
        max_retries: int = Field(default=3, env="MAX_RETRIES")

        # External APIs (Optional)
        alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
        polygon_api_key: Optional[str] = Field(default=None, env="POLYGON_API_KEY")
        finnhub_api_key: Optional[str] = Field(default=None, env="FINNHUB_API_KEY")

        # Logging
        log_level: str = Field(default="INFO", env="LOG_LEVEL")
        log_file_path: Path = Field(default=Path("./logs/app.log"), env="LOG_FILE_PATH")
        log_max_size: str = Field(default="10MB", env="LOG_MAX_SIZE")
        log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")

        # Rate Limiting
        rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
        rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")  # seconds

        # ML Model Settings
        feature_importance_threshold: float = Field(
            default=0.01, env="FEATURE_IMPORTANCE_THRESHOLD"
        )
        cross_validation_folds: int = Field(default=5, env="CROSS_VALIDATION_FOLDS")
        hyperparameter_tuning: bool = Field(default=False, env="HYPERPARAMETER_TUNING")
        auto_retrain: bool = Field(default=False, env="AUTO_RETRAIN")
        retrain_interval: str = Field(default="24h", env="RETRAIN_INTERVAL")

        # Stock Market Settings
        market_timezone: str = Field(default="Asia/Kolkata", env="MARKET_TIMEZONE")
        trading_hours_start: str = Field(default="09:15", env="TRADING_HOURS_START")
        trading_hours_end: str = Field(default="15:30", env="TRADING_HOURS_END")
        weekend_trading: bool = Field(default=False, env="WEEKEND_TRADING")

        # Notification Settings (Optional)
        slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
        discord_webhook_url: Optional[str] = Field(default=None, env="DISCORD_WEBHOOK_URL")
        email_smtp_server: Optional[str] = Field(default=None, env="EMAIL_SMTP_SERVER")
        email_smtp_port: int = Field(default=587, env="EMAIL_SMTP_PORT")
        email_username: Optional[str] = Field(default=None, env="EMAIL_USERNAME")
        email_password: Optional[str] = Field(default=None, env="EMAIL_PASSWORD")

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Create directories if they don't exist
            self.model_save_path.mkdir(parents=True, exist_ok=True)
            self.data_cache_path.mkdir(parents=True, exist_ok=True)
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create settings instance with Pydantic
    settings = Settings()

except (ImportError, NameError):
    # Fallback is already created above
    pass

# Export settings
__all__ = ["settings"]


# Additional utility functions
def get_project_root() -> Path:
    """
    Get the project root directory.
    """
    return Path(__file__).parent


def is_development() -> bool:
    """
    Check if running in development mode.
    """
    return settings.environment.lower() in ["development", "dev", "local"]


def is_production() -> bool:
    """
    Check if running in production mode.
    """
    return settings.environment.lower() in ["production", "prod"]


def get_log_config() -> dict:
    """
    Get logging configuration.
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level,
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.log_level,
                "formatter": "detailed",
                "filename": str(settings.log_file_path),
                "maxBytes": 10485760,  # 10MB
                "backupCount": settings.log_backup_count,
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["console", "file"],
        },
    }


# Environment-specific configurations
if is_development():
    # Development settings
    DEFAULT_SYMBOLS = ["TCS", "RELIANCE", "INFY", "HDFC", "ICICIBANK"]
else:
    # Production settings
    DEFAULT_SYMBOLS = [
        "TCS",
        "RELIANCE",
        "INFY",
        "HDFC",
        "ICICIBANK",
        "SBIN",
        "HINDUNILVR",
        "ITC",
        "KOTAKBANK",
        "BAJFINANCE",
        "ASIANPAINT",
        "MARUTI",
        "AXISBANK",
        "WIPRO",
        "ULTRACEMCO",
        "NESTLEIND",
        "HCLTECH",
        "LT",
        "SUNPHARMA",
        "TITAN",
    ]

# Stock market configuration
MARKET_CONFIG = {
    "NSE": {
        "timezone": "Asia/Kolkata",
        "trading_hours": {"start": "09:15", "end": "15:30"},
        "trading_days": [0, 1, 2, 3, 4],  # Monday to Friday
        "currency": "INR",
        "symbol_suffix": ".NS",
    }
}

# Model configuration
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
    },
    "gradient_boost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
    },
    "linear_regression": {"fit_intercept": True, "normalize": False},
}

# Feature configuration
FEATURE_CONFIG = {
    "technical_indicators": [
        "MA_5",
        "MA_10",
        "MA_20",
        "MA_50",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "BB_Upper",
        "BB_Lower",
        "BB_Middle",
        "ATR",
        "Stoch_K",
        "Williams_R",
    ],
    "price_features": [
        "Open",
        "High",
        "Low",
        "Volume",
        "Price_Change",
        "Price_Change_5",
        "HL_Spread",
        "Volume_Ratio",
        "Volatility",
    ],
    "min_data_points": 50,
    "max_features": 25,
}

# Cache configuration
CACHE_CONFIG = {
    "stock_data": {"ttl": 300, "key_prefix": "stock_data"},  # 5 minutes
    "analysis_results": {"ttl": 180, "key_prefix": "analysis"},  # 3 minutes
    "predictions": {"ttl": 60, "key_prefix": "prediction"},  # 1 minute
}

# API rate limiting configuration
RATE_LIMIT_CONFIG = {
    "default": {
        "requests": settings.rate_limit_requests,
        "period": settings.rate_limit_period,
    },
    "heavy_endpoints": {
        "requests": 10,
        "period": 60,  # 10 requests per minute for heavy operations
    },
    "batch_endpoints": {
        "requests": 5,
        "period": 300,  # 5 requests per 5 minutes for batch operations
    },
}

print(f"✅ Configuration loaded: {settings.environment} environment")
if is_development():
    print(f"📊 Debug mode: {settings.debug}")
    print(f"🌐 API will run on: http://{settings.api_host}:{settings.api_port}")
