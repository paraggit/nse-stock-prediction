"""
NSE Stock Prediction Modules Package.

This package contains the core modules for stock analysis:
- stock_data_fetcher: Fetches stock data from various sources
- technical_indicators: Calculates technical analysis indicators
- prediction_model: Machine learning models for price prediction
- stock_analyzer: Main orchestrator combining all components

"""

import sys
from pathlib import Path

# Add modules to Python path
modules_dir = Path(__file__).parent
if str(modules_dir) not in sys.path:
    sys.path.insert(0, str(modules_dir))

__version__ = "1.0.0"
__author__ = "NSE Stock Prediction Team"

# Import main components (with error handling)
try:
    from .prediction_model import StockPredictionModel
    from .stock_analyzer import StockAnalyzer
    from .stock_data_fetcher import StockDataFetcher
    from .technical_indicators import TechnicalIndicators

    __all__ = [
        "StockDataFetcher",
        "TechnicalIndicators",
        "StockPredictionModel",
        "StockAnalyzer",
    ]

except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")
    __all__ = []
