# NSE Stock Prediction Pipeline

A comprehensive machine learning pipeline for analyzing and predicting NSE (National Stock Exchange)
listed stocks with advanced technical indicators, multiple ML models, and API integration capabilities.
Built with Python Poetry for modern dependency management.

## 🚀 Features

- **Stock Data Fetching**: Real-time data from Yahoo Finance for NSE stocks
- **Technical Analysis**: 20+ technical indicators including RSI, MACD, Bollinger Bands, Moving Averages
- **ML Prediction Models**: Multiple algorithms (Random Forest, Gradient Boosting, Linear Regression)
- **API Ready**: Modular design for API integration
- **Caching System**: Intelligent caching for improved performance
- **Signal Generation**: Automated buy/sell signals based on technical analysis
- **Model Persistence**: Save and load trained models
- **Poetry Integration**: Modern dependency management with Poetry

## 📁 Project Structure

```
nse-stock-prediction/
├── README.md
├── pyproject.toml              # Poetry configuration
├── poetry.lock                 # Poetry lock file
├── .env.example               # Environment variables template
├── .gitignore
├── config.py
├── main.py
├── api/
│   ├── __init__.py
│   ├── routes.py
│   ├── models.py
│   └── dependencies.py
├── modules/
│   ├── __init__.py
│   ├── stock_data_fetcher.py
│   ├── technical_indicators.py
│   ├── prediction_model.py
│   └── stock_analyzer.py
├── models/
│   └── saved_models/
├── logs/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data_fetcher.py
│   ├── test_technical_indicators.py
│   └── test_prediction_model.py
└── scripts/
    ├── setup_project.py
    └── train_models.py
```

## 🛠️ Installation with Poetry

### Prerequisites

- Python 3.8+
- Poetry (latest version)

### Install Poetry

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Or using pip
pip install poetry

# Verify installation
poetry --version
```

### Project Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-username/nse-stock-prediction.git
cd nse-stock-prediction
```

2. **Install dependencies using Poetry**

```bash
# Install all dependencies (including dev dependencies)
poetry install

# Install only production dependencies
poetry install --only=main

# Install with specific groups
poetry install --with dev,test
```

3. **Activate the virtual environment**

```bash
# Activate Poetry shell
poetry shell

# Or run commands directly
poetry run python main.py
```

## 📦 Poetry Configuration

### `pyproject.toml`

```toml
[tool.poetry]
name = "nse-stock-prediction"
version = "1.0.0"
description = "Advanced NSE stock prediction pipeline with ML and technical analysis"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"
packages = [{include = "modules"}, {include = "api"}]

[tool.poetry.dependencies]
python = "^3.8"
yfinance = "^0.2.18"
pandas = "^2.0.0"
numpy = "^1.24.0"
scikit-learn = "^1.3.0"
matplotlib = "^3.7.0"
seaborn = "^0.12.0"
ta = "^0.10.2"
fastapi = "^0.104.0"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
pydantic = "^2.4.0"
joblib = "^1.3.0"
python-multipart = "^0.0.6"
aiofiles = "^23.0.0"
python-dotenv = "^1.0.0"
loguru = "^0.7.0"
httpx = "^0.25.0"
rich = "^13.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
pytest-asyncio = "^0.21.0"
black = "^23.0.0"
isort = "^5.12.0"
flake8 = "^6.0.0"
mypy = "^1.5.0"
pre-commit = "^3.4.0"
jupyter = "^1.0.0"
ipykernel = "^6.25.0"

[tool.poetry.group.test.dependencies]
pytest-mock = "^3.11.0"
factory-boy = "^3.3.0"
freezegun = "^1.2.0"
responses = "^0.23.0"

[tool.poetry.group.docs.dependencies]
mkdocs = "^1.5.0"
mkdocs-material = "^9.2.0"
mkdocstrings = {extras = ["python"], version = "^0.22.0"}

[tool.poetry.scripts]
stock-api = "main:main"
train-models = "scripts.train_models:main"
setup-project = "scripts.setup_project:main"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 100
target-version = ['py38']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 100
known_first_party = ["modules", "api"]

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers --strict-config"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests"
]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "yfinance.*",
    "ta.*",
    "matplotlib.*",
    "seaborn.*"
]
ignore_missing_imports = true

[tool.coverage.run]
source = ["modules", "api"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__init__.py"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError"
]
```

## 🚀 Quick Start with Poetry

### Development Setup

```bash
# Clone and setup
git clone https://github.com/your-username/nse-stock-prediction.git
cd nse-stock-prediction

# Install dependencies
poetry install

# Setup pre-commit hooks
poetry run pre-commit install

# Copy environment file
cp .env.example .env

# Run setup script
poetry run setup-project
```

### Basic Usage

```bash
# Activate Poetry shell
poetry shell

# Run the application
python main.py

# Or run directly with Poetry
poetry run python main.py

# Use custom scripts
poetry run stock-api
poetry run train-models
```

### Development Commands

```bash
# Code formatting
poetry run black .
poetry run isort .

# Linting
poetry run flake8 .
poetry run mypy .

# Testing
poetry run pytest
poetry run pytest --cov=modules --cov=api
poetry run pytest -m "not slow"

# Run specific tests
poetry run pytest tests/test_prediction_model.py -v
```

## 📊 Poetry Scripts and Commands

### Custom Scripts (defined in pyproject.TOML)

```bash
# Start the API server
poetry run stock-api

# Train models for multiple stocks
poetry run train-models

# Setup project directories and configs
poetry run setup-project
```

### Development Workflow

```bash
# 1. Add new dependency
poetry add requests

# 2. Add development dependency
poetry add --group dev pytest-mock

# 3. Update dependencies
poetry update

# 4. Show dependency tree
poetry show --tree

# 5. Check for outdated packages
poetry show --outdated

# 6. Export requirements.txt (if needed)
poetry export -f requirements.txt --output requirements.txt

# 7. Build the package
poetry build

# 8. Publish to PyPI (if applicable)
poetry publish
```

## 🐳 Docker Integration with Poetry

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Install Poetry
RUN pip install poetry

# Set work directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Configure Poetry: Don't create virtual env, install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-interaction --no-ansi

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["poetry", "run", "stock-api"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: "3.8"

services:
  stock-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    command: poetry run uvicorn main:app --host 0.0.0.0 --port 8000

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## 🧪 Testing with Poetry

### Test Configuration

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=modules --cov=api --cov-report=html

# Run specific test categories
poetry run pytest -m unit
poetry run pytest -m integration
poetry run pytest -m "not slow"

# Run tests in parallel
poetry run pytest -n auto

# Generate coverage report
poetry run pytest --cov=modules --cov-report=term-missing
```

### Test Structure

```python
# tests/conftest.py
import pytest
from modules.stock_analyzer import StockAnalyzer

@pytest.fixture
def analyzer():
    return StockAnalyzer(model_type='random_forest')

@pytest.fixture
def sample_stock_data():
    # Return sample data for testing
    pass

# tests/test_stock_analyzer.py
import pytest

@pytest.mark.unit
def test_analyzer_initialization(analyzer):
    assert analyzer.model_type == 'random_forest'

@pytest.mark.integration
@pytest.mark.slow
def test_full_analysis_pipeline(analyzer):
    result = analyzer.analyze_stock('TCS', period='1mo')
    assert result['status'] == 'success'
```

## 📚 Poetry Dependency Groups

### Main Dependencies

```bash
# Add production dependency
poetry add pandas numpy scikit-learn

# Add with extras
poetry add "uvicorn[standard]"
```

### Development Dependencies

```bash
# Add to dev group
poetry add --group dev black isort flake8 mypy

# Add to test group
poetry add --group test pytest pytest-cov factory-boy

# Add to docs group
poetry add --group docs mkdocs mkdocs-material
```

### Installing Specific Groups

```bash
# Install only main dependencies
poetry install --only=main

# Install main + dev
poetry install --with dev

# Install everything except docs
poetry install --without docs
```

## 🔧 Environment Configuration

### `.env.example`

```bash
# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Model Configuration
DEFAULT_MODEL_TYPE=random_forest
CACHE_TIMEOUT=300

# External APIs (optional)
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
```

### Configuration Loading

```python
# config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    default_model_type: str = "random_forest"
    cache_timeout: int = 300

    alpha_vantage_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

## 📈 Usage Examples with Poetry

### 1. Development Workflow

```bash
# Start development session
poetry shell

# Run with auto-reload
poetry run uvicorn main:app --reload

# Format code before commit
poetry run black . && poetry run isort .

# Run tests
poetry run pytest -v
```

### 2. Production Deployment

```bash
# Install production dependencies only
poetry install --only=main

# Build optimized package
poetry build

# Run production server
poetry run gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 3. CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Poetry
        uses: snok/install-poetry@v1
        with:
          virtualenvs-create: true
          virtualenvs-in-project: true

      - name: Load cached venv
        id: cached-poetry-dependencies
        uses: actions/cache@v3
        with:
          path: .venv
          key: venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('**/poetry.lock') }}

      - name: Install dependencies
        if: steps.cached-poetry-dependencies.outputs.cache-hit != 'true'
        run: poetry install --no-interaction --no-root

      - name: Install project
        run: poetry install --no-interaction

      - name: Run tests
        run: |
          poetry run pytest --cov=modules --cov=api --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## 🚀 Deployment Options

### 1. Direct Deployment

```bash
# Production install
poetry install --only=main

# Run with Gunicorn
poetry run gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 2. Docker Deployment

```bash
# Build image
docker build -t nse-stock-prediction .

# Run container
docker run -p 8000:8000 nse-stock-prediction
```

### 3. Poetry Build and Install

```bash
# Build wheel
poetry build

# Install built package
pip install dist/nse_stock_prediction-1.0.0-py3-none-any.whl
```

## 📊 Performance and Monitoring

### Dependency Analysis

```bash
# Check dependency sizes
poetry show --tree

# Security audit
poetry audit

# Check for updates
poetry show --outdated

# Update specific package
poetry update pandas

# Lock file verification
poetry lock --check
```

## 🤝 Contributing with Poetry

### Development Setup for Contributors

```bash
# Fork and clone
git clone https://github.com/your-username/nse-stock-prediction.git
cd nse-stock-prediction

# Install with all groups
poetry install --with dev,test,docs

# Setup pre-commit
poetry run pre-commit install

# Create feature branch
git checkout -b feature/new-feature

# Make changes and test
poetry run pytest
poetry run black .
poetry run isort .

# Commit and push
git commit -m "Add new feature"
git push origin feature/new-feature
```

### Adding New Dependencies

```bash
# Add runtime dependency
poetry add new-package

# Add development dependency
poetry add --group dev new-dev-package

# Add optional dependency
poetry add --optional new-optional-package

# Update lock file
poetry lock --no-update
```

## ⚡ Performance Tips

1. **Use Poetry Groups**: Organize dependencies by purpose
2. **Lock File**: Always commit `poetry.lock` for reproducible builds
3. **Virtual Environments**: Let Poetry manage virtual environments
4. **Caching**: Use dependency caching in CI/CD
5. **Build Optimization**: Use `--only=main` for production builds

## 🔍 Troubleshooting

### Common Poetry Issues

```bash
# Clear cache
poetry cache clear pypi --all

# Rebuild lock file
rm poetry.lock && poetry lock

# Fix virtual environment issues
poetry env remove python
poetry install

# Check configuration
poetry config --list

# Verbose installation
poetry install -vvv
```

This documentation provides a complete guide for using the NSE Stock Prediction Pipeline with Poetry,
including modern Python development practices, dependency management, and deployment strategies.
