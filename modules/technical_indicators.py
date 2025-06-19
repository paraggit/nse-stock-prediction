"""
Simple Technical Indicators Module for stock analysis.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Simple technical indicators calculator.
    """

    def __init__(self):
        self.indicators = {}

    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate simple moving averages.

        Args:
            data: Stock price data

        Returns:
            DataFrame with moving averages

        """
        df = data.copy()

        # Simple Moving Averages
        df["MA_5"] = df["Close"].rolling(window=5).mean()
        df["MA_10"] = df["Close"].rolling(window=10).mean()
        df["MA_20"] = df["Close"].rolling(window=20).mean()
        df["MA_50"] = df["Close"].rolling(window=50).mean()

        return df

    def calculate_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Calculate RSI (Relative Strength Index)

        Args:
            data: Stock price data
            window: RSI calculation window

        Returns:
            DataFrame with RSI

        """
        df = data.copy()

        # Calculate price changes
        delta = df["Close"].diff()

        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        # Calculate RSI
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        return df

    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Args:
            data: Stock price data

        Returns:
            DataFrame with MACD

        """
        df = data.copy()

        # Calculate EMAs
        ema_12 = df["Close"].ewm(span=12).mean()
        ema_26 = df["Close"].ewm(span=26).mean()

        # MACD line
        df["MACD"] = ema_12 - ema_26

        # Signal line
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

        # Histogram
        df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

        return df

    def calculate_bollinger_bands(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Args:
            data: Stock price data
            window: Calculation window

        Returns:
            DataFrame with Bollinger Bands

        """
        df = data.copy()

        # Calculate SMA and standard deviation
        sma = df["Close"].rolling(window=window).mean()
        std = df["Close"].rolling(window=window).std()

        # Bollinger Bands
        df["BB_Middle"] = sma
        df["BB_Upper"] = sma + (std * 2)
        df["BB_Lower"] = sma - (std * 2)

        return df

    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators.

        Args:
            data: Stock price data

        Returns:
            DataFrame with volume indicators

        """
        df = data.copy()

        # Volume Moving Average
        df["Volume_MA"] = df["Volume"].rolling(window=10).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]

        return df

    def calculate_price_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price pattern indicators.

        Args:
            data: Stock price data

        Returns:
            DataFrame with price pattern indicators

        """
        df = data.copy()

        # Price changes
        df["Price_Change"] = df["Close"].pct_change()
        df["Price_Change_5"] = df["Close"].pct_change(periods=5)

        # High-Low relationships
        df["HL_Spread"] = (df["High"] - df["Low"]) / df["Close"]

        # Volatility
        df["Volatility"] = df["Close"].rolling(window=20).std()

        return df

    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            data: Stock price data

        Returns:
            DataFrame with all technical indicators

        """
        try:
            df = data.copy()

            # Apply all indicator calculations
            df = self.calculate_moving_averages(df)
            df = self.calculate_rsi(df)
            df = self.calculate_macd(df)
            df = self.calculate_bollinger_bands(df)
            df = self.calculate_volume_indicators(df)
            df = self.calculate_price_patterns(df)

            # Add simple technical indicators if ta library is available
            try:
                import ta

                # Additional indicators with ta library
                df["ATR"] = ta.volatility.AverageTrueRange(
                    df["High"], df["Low"], df["Close"]
                ).average_true_range()
                df["Stoch_K"] = ta.momentum.StochasticOscillator(
                    df["High"], df["Low"], df["Close"]
                ).stoch()
                df["Williams_R"] = ta.momentum.WilliamsRIndicator(
                    df["High"], df["Low"], df["Close"]
                ).williams_r()

            except ImportError:
                logger.warning("TA library not available, using basic indicators only")
                # Use simple alternatives
                df["ATR"] = df["HL_Spread"] * df["Close"]  # Simple ATR approximation
                df["Stoch_K"] = np.nan  # Will be filled with NaN
                df["Williams_R"] = np.nan

            logger.info("Successfully calculated all technical indicators")
            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return data

    def get_latest_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get latest trading signals based on technical indicators.

        Args:
            data: Stock data with technical indicators

        Returns:
            Dictionary with trading signals

        """
        try:
            if data.empty:
                return {"error": "No data available for signal generation"}

            latest = data.iloc[-1]

            # RSI Signal
            rsi_signal = "Neutral"
            if "RSI" in latest and pd.notna(latest["RSI"]):
                if latest["RSI"] < 30:
                    rsi_signal = "Oversold"
                elif latest["RSI"] > 70:
                    rsi_signal = "Overbought"

            # MACD Signal
            macd_signal = "Neutral"
            if "MACD" in latest and "MACD_Signal" in latest:
                if pd.notna(latest["MACD"]) and pd.notna(latest["MACD_Signal"]):
                    if latest["MACD"] > latest["MACD_Signal"]:
                        macd_signal = "Bullish"
                    else:
                        macd_signal = "Bearish"

            # Moving Average Signal
            ma_signal = "Neutral"
            if "MA_20" in latest and pd.notna(latest["MA_20"]):
                if latest["Close"] > latest["MA_20"]:
                    ma_signal = "Bullish"
                else:
                    ma_signal = "Bearish"

            # Bollinger Bands Signal
            bb_signal = "Neutral"
            if all(col in latest for col in ["BB_Upper", "BB_Lower"]) and all(
                pd.notna(latest[col]) for col in ["BB_Upper", "BB_Lower"]
            ):
                if latest["Close"] > latest["BB_Upper"]:
                    bb_signal = "Overbought"
                elif latest["Close"] < latest["BB_Lower"]:
                    bb_signal = "Oversold"

            # Volume Signal
            volume_signal = "Normal Volume"
            if "Volume_Ratio" in latest and pd.notna(latest["Volume_Ratio"]):
                if latest["Volume_Ratio"] > 1.5:
                    volume_signal = "High Volume"
                elif latest["Volume_Ratio"] < 0.5:
                    volume_signal = "Low Volume"

            # Overall Signal
            bullish_signals = [
                rsi_signal == "Oversold",
                macd_signal == "Bullish",
                ma_signal == "Bullish",
            ]
            bearish_signals = [
                rsi_signal == "Overbought",
                macd_signal == "Bearish",
                ma_signal == "Bearish",
            ]

            bullish_count = sum(bullish_signals)
            bearish_count = sum(bearish_signals)

            if bullish_count > bearish_count:
                overall_signal = "Bullish"
            elif bearish_count > bullish_count:
                overall_signal = "Bearish"
            else:
                overall_signal = "Neutral"

            return {
                "rsi_signal": rsi_signal,
                "macd_signal": macd_signal,
                "ma_signal": ma_signal,
                "bb_signal": bb_signal,
                "volume_signal": volume_signal,
                "overall_signal": overall_signal,
            }

        except Exception as e:
            logger.error(f"Error getting latest signals: {str(e)}")
            return {"error": str(e)}
