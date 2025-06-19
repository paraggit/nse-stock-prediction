"""
NSE Stock Prediction API Package.

This package provides a RESTful API for stock analysis and prediction
using machine learning models and technical analysis.

Main Components:
- FastAPI application with automatic OpenAPI documentation
- Stock analysis endpoints with caching
- Real-time prediction services
- Technical analysis tools
- Model management endpoints
- Health monitoring and metrics

"""

# API metadata
API_TITLE = "NSE Stock Prediction API"
API_DESCRIPTION = """
🚀 **Advanced NSE Stock Prediction Pipeline**

A comprehensive API for analyzing and predicting Indian stock market (NSE) movements using:

- **Machine Learning Models**: Random Forest, Gradient Boosting, Linear Regression
- **Technical Analysis**: 20+ indicators including RSI, MACD, Bollinger Bands
- **Real-time Data**: Live stock prices and market data
- **Historical Analysis**: Trend analysis and pattern recognition
- **Portfolio Tools**: Multi-stock comparison and analysis

## Features

### 📊 Stock Analysis
- Complete stock analysis with predictions and technical signals
- Historical data analysis with customizable periods
- Real-time price updates and market status

### 🤖 Machine Learning
- Multiple ML models for price prediction
- Feature importance analysis
- Model performance metrics and validation
- Automated model retraining capabilities

### 📈 Technical Analysis
- 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Trading signal generation
- Support and resistance level detection
- Volume analysis and momentum indicators

### 🛠️ API Management
- RESTful endpoints with OpenAPI documentation
- Request/response validation with Pydantic
- Rate limiting and caching for performance
- Health monitoring and metrics
- Error handling and logging

## Quick Start

```python
import requests

# Get stock analysis
response = requests.get("http://localhost:8000/api/v1/stocks/TCS/analyze")
data = response.json()

print(f"Current Price: ₹{data['current_price']}")
print(f"Predicted Price: ₹{data['prediction']['predicted_price']}")
print(f"Technical Signal: {data['technical_signals']['overall_signal']}")
```

## Popular NSE Stocks Supported

**IT Sector**: TCS, INFY, WIPRO, HCLTECH, TECHM
**Banking**: HDFC, ICICIBANK, SBIN, KOTAKBANK, AXISBANK
**Energy**: RELIANCE, ONGC, BPCL, IOC
**Auto**: TATAMOTORS, MARUTI, BAJAJ-AUTO, M&M
**FMCG**: HINDUNILVR, ITC, NESTLEIND, BRITANNIA

## Data Sources

- **Yahoo Finance**: Real-time and historical stock data
- **Technical Analysis Library**: Advanced indicator calculations
- **NSE**: Market hours and trading calendar information

---

⚠️ **Disclaimer**: This API is for educational and research purposes only.
Stock predictions are not guaranteed and should not be used as sole investment advice.
Always consult with financial advisors before making investment decisions.
"""

API_VERSION = "1.0.0"
API_CONTACT = {
    "name": "NSE Stock Prediction API Support",
    "url": "https://github.com/your-username/nse-stock-prediction",
    "email": "support@example.com",
}

API_LICENSE = {"name": "MIT License", "url": "https://opensource.org/licenses/MIT"}

API_TAGS_METADATA = [
    {
        "name": "stocks",
        "description": "Stock analysis and prediction endpoints. Get comprehensive analysis including price predictions, technical indicators, and trading signals.",
        "externalDocs": {
            "description": "Stock Analysis Documentation",
            "url": "https://github.com/your-username/nse-stock-prediction#stock-analysis",
        },
    },
    {
        "name": "technical",
        "description": "Technical analysis endpoints. Access 20+ technical indicators, trading signals, and market momentum analysis.",
        "externalDocs": {
            "description": "Technical Analysis Guide",
            "url": "https://github.com/your-username/nse-stock-prediction#technical-analysis",
        },
    },
    {
        "name": "predictions",
        "description": "Machine learning prediction endpoints. Get AI-powered price predictions with confidence intervals and model metrics.",
    },
    {
        "name": "models",
        "description": "Model management endpoints. Train, save, load, and compare different machine learning models.",
    },
    {
        "name": "market",
        "description": "Market data and status endpoints. Get market hours, trading status, and real-time market information.",
    },
    {
        "name": "portfolio",
        "description": "Portfolio analysis endpoints. Compare multiple stocks and analyze portfolio performance.",
    },
    {
        "name": "health",
        "description": "System health and monitoring endpoints. Check API status, performance metrics, and system diagnostics.",
    },
]


# Conditional imports to avoid circular dependencies
def create_app():
    """
    Create FastAPI application (imported on demand).
    """
    try:
        from .main import create_app as _create_app

        return _create_app()
    except ImportError as e:
        print(f"Warning: Could not import FastAPI app: {e}")
        return None


def get_app():
    """
    Get FastAPI application instance (imported on demand).
    """
    try:
        from .main import app

        return app
    except ImportError as e:
        print(f"Warning: Could not import FastAPI app instance: {e}")
        return None


# Export main components
__all__ = [
    "create_app",
    "get_app",
    "API_TITLE",
    "API_DESCRIPTION",
    "API_VERSION",
    "API_CONTACT",
    "API_LICENSE",
    "API_TAGS_METADATA",
]

__version__ = API_VERSION
