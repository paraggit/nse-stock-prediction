# scripts/setup_project.py
"""
Project setup script for NSE Stock Prediction Pipeline
Run with: poetry run setup-project
"""

import os
import sys
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.progress import Progress, TaskID

app = typer.Typer()
console = Console()


def create_directories() -> List[Path]:
    """
    Create necessary project directories.
    """
    directories = [
        Path("models/saved_models"),
        Path("logs"),
        Path("data/cache"),
        Path("data/raw"),
        Path("data/processed"),
        Path("notebooks"),
        Path("docs"),
        Path("scripts"),
        Path("tests/fixtures"),
        Path("api/routers"),
        Path("modules/utils"),
    ]

    created_dirs = []
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)

    return created_dirs


def create_gitignore() -> None:
    """
    Create comprehensive .gitignore file.
    """
    gitignore_content = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock

# pdm
.pdm.toml

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
.idea/

# VSCode
.vscode/

# Project specific
models/saved_models/*.pkl
models/saved_models/*.joblib
logs/*.log
data/cache/*
data/raw/*
data/processed/*
!data/cache/.gitkeep
!data/raw/.gitkeep
!data/processed/.gitkeep

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Temporary files
*.tmp
*.temp
*~

# API keys and secrets
.secrets
api_keys.json
"""

    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        gitignore_path.write_text(gitignore_content.strip())
        console.print("✅ Created .gitignore file", style="green")


def create_env_example() -> None:
    """
    Create .env.example file with all environment variables.
    """
    env_content = """
# Application Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
APP_NAME=NSE Stock Prediction API
APP_VERSION=1.0.0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_RELOAD=true
API_DOCS_URL=/docs
API_REDOC_URL=/redoc

# Database Configuration (Optional)
DATABASE_URL=sqlite:///./stock_data.db
POSTGRES_USER=stock_user
POSTGRES_PASSWORD=stock_password
POSTGRES_DB=stock_prediction
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Redis Configuration (Optional)
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL=300

# Model Configuration
DEFAULT_MODEL_TYPE=random_forest
MODEL_SAVE_PATH=./models/saved_models
PREDICTION_CACHE_TTL=60
BATCH_SIZE=32
MAX_WORKERS=4

# Data Configuration
DATA_CACHE_PATH=./data/cache
DATA_FETCH_TIMEOUT=30
YAHOO_FINANCE_DELAY=1
MAX_RETRIES=3

# External API Keys (Optional)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
POLYGON_API_KEY=your_polygon_key_here
FINNHUB_API_KEY=your_finnhub_key_here
QUANDL_API_KEY=your_quandl_key_here

# Monitoring and Logging
SENTRY_DSN=your_sentry_dsn_here
LOG_FILE_PATH=./logs/app.log
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5

# Security
SECRET_KEY=your-secret-key-here-change-in-production
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60

# ML Model Settings
FEATURE_IMPORTANCE_THRESHOLD=0.01
CROSS_VALIDATION_FOLDS=5
HYPERPARAMETER_TUNING=false
AUTO_RETRAIN=false
RETRAIN_INTERVAL=24h

# Stock Market Settings
MARKET_TIMEZONE=Asia/Kolkata
TRADING_HOURS_START=09:15
TRADING_HOURS_END=15:30
WEEKEND_TRADING=false

# Notification Settings (Optional)
SLACK_WEBHOOK_URL=your_slack_webhook_here
DISCORD_WEBHOOK_URL=your_discord_webhook_here
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
"""

    env_example_path = Path(".env.example")
    if not env_example_path.exists():
        env_example_path.write_text(env_content.strip())
        console.print("✅ Created .env.example file", style="green")


def create_gitkeep_files(directories: List[Path]) -> None:
    """
    Create .gitkeep files in empty directories.
    """
    for directory in directories:
        if directory.name in ["cache", "raw", "processed", "saved_models"]:
            gitkeep_file = directory / ".gitkeep"
            if not gitkeep_file.exists():
                gitkeep_file.touch()


def setup_pre_commit() -> None:
    """
    Setup pre-commit configuration.
    """
    pre_commit_config = """
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-merge-conflict
      - id: check-toml
      - id: debug-statements
      - id: name-tests-test

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.287
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-PyYAML]
        exclude: ^(docs/|tests/)

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: [-c, pyproject.toml]
        additional_dependencies: ["bandit[toml]"]

  - repo: https://github.com/python-poetry/poetry
    rev: 1.6.0
    hooks:
      - id: poetry-check
      - id: poetry-lock
        args: [--no-update]
"""

    pre_commit_path = Path(".pre-commit-config.yaml")
    if not pre_commit_path.exists():
        pre_commit_path.write_text(pre_commit_config.strip())
        console.print("✅ Created .pre-commit-config.yaml file", style="green")


def create_makefile() -> None:
    """
    Create Makefile for common development tasks.
    """
    makefile_content = """
.PHONY: help install install-dev test test-cov lint format clean run docker-build docker-run

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	poetry install --only=main

install-dev: ## Install all dependencies including dev
	poetry install --with dev,test,docs

test: ## Run tests
	poetry run pytest

test-cov: ## Run tests with coverage
	poetry run pytest --cov=modules --cov=api --cov-report=html --cov-report=term

lint: ## Run linting
	poetry run black --check .
	poetry run isort --check-only .
	poetry run ruff check .
	poetry run mypy .

format: ## Format code
	poetry run black .
	poetry run isort .
	poetry run ruff check . --fix

clean: ## Clean cache and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

run: ## Run the application
	poetry run python main.py

run-dev: ## Run the application in development mode
	poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000

setup: ## Setup the project (run once after cloning)
	poetry install --with dev,test,docs
	poetry run pre-commit install
	poetry run setup-project

build: ## Build the package
	poetry build

publish: ## Publish to PyPI (requires authentication)
	poetry publish

docker-build: ## Build Docker image
	docker build -t nse-stock-prediction .

docker-run: ## Run Docker container
	docker run -p 8000:8000 nse-stock-prediction

update: ## Update dependencies
	poetry update

lock: ## Update lock file
	poetry lock --no-update

check: ## Check project health
	poetry check
	poetry run safety check
	poetry run bandit -r modules/ api/

docs-serve: ## Serve documentation locally
	poetry run mkdocs serve

docs-build: ## Build documentation
	poetry run mkdocs build

benchmark: ## Run performance benchmarks
	poetry run pytest tests/ -m benchmark --benchmark-only

all-checks: format lint test ## Run all checks (format, lint, test)
"""

    makefile_path = Path("Makefile")
    if not makefile_path.exists():
        makefile_path.write_text(makefile_content.strip())
        console.print("✅ Created Makefile", style="green")


@app.command()
def main(
    create_dirs: bool = typer.Option(True, help="Create project directories"),
    create_configs: bool = typer.Option(True, help="Create configuration files"),
    setup_git: bool = typer.Option(True, help="Setup Git configuration"),
    install_hooks: bool = typer.Option(True, help="Install pre-commit hooks"),
) -> None:
    """
    Setup the NSE Stock Prediction project.
    """

    console.print("\n🚀 Setting up NSE Stock Prediction Pipeline...\n", style="bold blue")

    with Progress() as progress:
        task = progress.add_task("Setting up project...", total=6)

        if create_dirs:
            console.print("📁 Creating project directories...")
            created_dirs = create_directories()
            create_gitkeep_files(created_dirs)
            console.print(f"✅ Created {len(created_dirs)} directories", style="green")
            progress.advance(task)

        if create_configs:
            console.print("⚙️  Creating configuration files...")
            create_env_example()
            create_makefile()
            progress.advance(task)

        if setup_git:
            console.print("🔧 Setting up Git configuration...")
            create_gitignore()
            setup_pre_commit()
            progress.advance(task)

        if install_hooks:
            console.print("🪝 Installing pre-commit hooks...")
            try:
                os.system("poetry run pre-commit install")
                console.print("✅ Pre-commit hooks installed", style="green")
            except Exception as e:
                console.print(f"⚠️  Could not install pre-commit hooks: {e}", style="yellow")
            progress.advance(task)

        console.print("📋 Creating initial README sections...")
        progress.advance(task)

        console.print("🎯 Final setup steps...")
        progress.advance(task)

    console.print("\n✨ Project setup complete! Next steps:\n", style="bold green")
    console.print("1. Copy .env.example to .env and configure your settings")
    console.print("2. Run 'poetry install --with dev,test,docs' to install dependencies")
    console.print("3. Run 'make test' to verify everything works")
    console.print("4. Start developing with 'make run-dev'\n")


if __name__ == "__main__":
    app()


import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console

# from rich.progress import Progress, TaskID
from rich.table import Table

# Import your modules
from modules.stock_analyzer import StockAnalyzer

app = typer.Typer()
console = Console()

# Popular NSE stocks for training
DEFAULT_STOCKS = [
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

SECTORS = {
    "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
    "BANKING": ["HDFC", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA"],
    "AUTO": ["MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "M&M"],
    "PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN"],
}


def train_single_stock(
    symbol: str,
    model_type: str = "random_forest",
    period: str = "2y",
    save_model: bool = True,
) -> Dict:
    """
    Train model for a single stock.
    """
    try:
        analyzer = StockAnalyzer(model_type=model_type)

        console.print(f"Training {model_type} model for {symbol}...", style="blue")

        start_time = time.time()
        result = analyzer.analyze_stock(
            symbol=symbol,
            period=period,
            include_prediction=True,
            include_technical=True,
        )
        training_time = time.time() - start_time

        if result.get("status") == "success":
            if save_model:
                model_path = Path(f"models/saved_models/{symbol}_{model_type}.joblib")
                analyzer.prediction_model.save_model(str(model_path))

            return {
                "symbol": symbol,
                "status": "success",
                "accuracy": result.get("model_metrics", {}).get("accuracy", 0),
                "r2_score": result.get("model_metrics", {}).get("test_r2", 0),
                "training_time": round(training_time, 2),
                "data_points": result.get("data_points", 0),
                "model_type": model_type,
            }
        else:
            return {
                "symbol": symbol,
                "status": "failed",
                "error": result.get("error", "Unknown error"),
                "model_type": model_type,
            }

    except Exception as e:
        return {
            "symbol": symbol,
            "status": "failed",
            "error": str(e),
            "model_type": model_type,
        }


@app.command()
def train_all(
    stocks: Optional[List[str]] = typer.Option(None, help="List of stock symbols"),
    model_type: str = typer.Option("random_forest", help="Model type to train"),
    period: str = typer.Option("2y", help="Data period for training"),
    sector: Optional[str] = typer.Option(None, help="Train stocks from specific sector"),
    save_models: bool = typer.Option(True, help="Save trained models"),
    parallel: bool = typer.Option(False, help="Train models in parallel"),
    output_file: Optional[str] = typer.Option(None, help="Save results to JSON file"),
) -> None:
    """
    Train models for multiple stocks.
    """

    # Determine which stocks to train
    if sector and sector.upper() in SECTORS:
        stock_list = SECTORS[sector.upper()]
        console.print(f"Training models for {sector.upper()} sector stocks", style="blue")
    elif stocks:
        stock_list = stocks
    else:
        stock_list = DEFAULT_STOCKS[:10]  # Limit to first 10 for demo
        console.print("Training models for default stock list", style="blue")

    console.print(f"Model type: {model_type}")
    console.print(f"Data period: {period}")
    console.print(f"Stocks to train: {len(stock_list)}")
    console.print(f"Stocks: {', '.join(stock_list)}\n")

    results = []

    with Progress() as progress:
        task = progress.add_task("Training models...", total=len(stock_list))

        for stock in stock_list:
            result = train_single_stock(
                symbol=stock,
                model_type=model_type,
                period=period,
                save_model=save_models,
            )
            results.append(result)
            progress.advance(task)

            # Print immediate result
            if result["status"] == "success":
                console.print(
                    f"✅ {stock}: Accuracy {result['accuracy']:.1f}% "
                    f"(R² {result['r2_score']:.3f}) in {result['training_time']}s",
                    style="green",
                )
            else:
                console.print(f"❌ {stock}: {result['error']}", style="red")

    # Summary table
    console.print("\n📊 Training Summary:", style="bold")

    table = Table()
    table.add_column("Symbol", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Accuracy", justify="right")
    table.add_column("R² Score", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Data Points", justify="right")

    successful = 0
    total_accuracy = 0

    for result in results:
        if result["status"] == "success":
            successful += 1
            total_accuracy += result["accuracy"]

            table.add_row(
                result["symbol"],
                "✅ Success",
                f"{result['accuracy']:.1f}%",
                f"{result['r2_score']:.3f}",
                f"{result['training_time']}",
                f"{result['data_points']}",
            )
        else:
            table.add_row(result["symbol"], "❌ Failed", "-", "-", "-", "-")

    console.print(table)

    # Statistics
    if successful > 0:
        avg_accuracy = total_accuracy / successful
        console.print(f"\n📈 Results: {successful}/{len(stock_list)} successful")
        console.print(f"📊 Average accuracy: {avg_accuracy:.1f}%")

    # Save results to file
    if output_file:
        output_path = Path(output_file)
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "period": period,
            "total_stocks": len(stock_list),
            "successful_trainings": successful,
            "average_accuracy": avg_accuracy if successful > 0 else 0,
            "results": results,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        console.print(f"💾 Results saved to {output_file}", style="green")


@app.command()
def train_single(
    symbol: str = typer.Argument(..., help="Stock symbol to train"),
    model_type: str = typer.Option("random_forest", help="Model type"),
    period: str = typer.Option("2y", help="Data period"),
    save_model: bool = typer.Option(True, help="Save the trained model"),
) -> None:
    """
    Train model for a single stock.
    """

    console.print(f"🤖 Training {model_type} model for {symbol}", style="bold blue")

    result = train_single_stock(
        symbol=symbol, model_type=model_type, period=period, save_model=save_model
    )

    if result["status"] == "success":
        console.print("✅ Training successful!", style="green")
        console.print(f"📊 Accuracy: {result['accuracy']:.1f}%")
        console.print(f"📈 R² Score: {result['r2_score']:.3f}")
        console.print(f"⏱️  Training time: {result['training_time']}s")
        console.print(f"📋 Data points: {result['data_points']}")

        if save_model:
            console.print(f"💾 Model saved for {symbol}", style="green")
    else:
        console.print(f"❌ Training failed: {result['error']}", style="red")


@app.command()
def compare_models(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    period: str = typer.Option("1y", help="Data period"),
    models: List[str] = typer.Option(["random_forest", "gradient_boost"], help="Models to compare"),
) -> None:
    """
    Compare different models for a single stock.
    """

    console.print(f"🔄 Comparing models for {symbol}", style="bold blue")

    results = []

    with Progress() as progress:
        task = progress.add_task("Training models...", total=len(models))

        for model_type in models:
            result = train_single_stock(
                symbol=symbol, model_type=model_type, period=period, save_model=False
            )
            results.append(result)
            progress.advance(task)

    # Comparison table
    table = Table(title=f"Model Comparison for {symbol}")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("R² Score", justify="right")
    table.add_column("Training Time", justify="right")
    table.add_column("Status", style="magenta")

    for result in results:
        if result["status"] == "success":
            table.add_row(
                result["model_type"],
                f"{result['accuracy']:.1f}%",
                f"{result['r2_score']:.3f}",
                f"{result['training_time']}s",
                "✅ Success",
            )
        else:
            table.add_row(result["model_type"], "-", "-", "-", "❌ Failed")

    console.print(table)

    # Best model
    successful_results = [r for r in results if r["status"] == "success"]
    if successful_results:
        best_model = max(successful_results, key=lambda x: x["accuracy"])
        console.print(
            f"\n🏆 Best model: {best_model['model_type']} "
            f"with {best_model['accuracy']:.1f}% accuracy",
            style="bold green",
        )


if __name__ == "__main__":
    app()
