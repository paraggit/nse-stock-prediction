"""
Simple Stock Data Fetcher Module for NSE stocks.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Simple stock data fetcher using yfinance.
    """

    def __init__(self):
        self.cache = {}

    def get_nse_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch NSE stock data using Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., 'TCS', 'RELIANCE')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')

        Returns:
            DataFrame with stock data

        """
        try:
            import yfinance as yf

            # Add .NS suffix for NSE stocks if not present
            if not symbol.endswith(".NS"):
                symbol += ".NS"

            logger.info(f"Fetching data for {symbol} with period {period}")

            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()

            logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
            return data

        except ImportError:
            logger.error("yfinance library not installed. Please install: pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed stock information.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with stock information

        """
        try:
            import yfinance as yf

            if not symbol.endswith(".NS"):
                symbol += ".NS"

            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "symbol": symbol,
                "company_name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "current_price": info.get("currentPrice", 0),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 0),
                "volume": info.get("volume", 0),
                "avg_volume": info.get("averageVolume", 0),
            }

        except ImportError:
            logger.error("yfinance library not installed")
            return {"symbol": symbol, "error": "yfinance not available"}
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return {"symbol": symbol, "error": str(e)}

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists.

        Args:
            symbol: Stock symbol to validate

        Returns:
            Boolean indicating if symbol is valid

        """
        try:
            data = self.get_nse_stock_data(symbol, period="5d")
            return not data.empty
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {str(e)}")
            return False
