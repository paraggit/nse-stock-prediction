"""
Main Stock Analyzer Module that combines all components.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .prediction_model import StockPredictionModel
from .stock_data_fetcher import StockDataFetcher
from .technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class StockAnalyzer:
    """
    Main class that orchestrates stock analysis pipeline.
    """

    def __init__(self, model_type: str = "random_forest"):
        self.data_fetcher = StockDataFetcher()
        self.technical_analyzer = TechnicalIndicators()
        self.prediction_model = StockPredictionModel(model_type)
        self.analysis_cache = {}
        self.model_type = model_type

    def analyze_stock(
        self,
        symbol: str,
        period: str = "1y",
        include_prediction: bool = True,
        include_technical: bool = True,
        include_info: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive stock analysis.

        Args:
            symbol: Stock symbol
            period: Data period
            include_prediction: Whether to include price prediction
            include_technical: Whether to include technical analysis
            include_info: Whether to include basic stock info

        Returns:
            Dictionary with complete analysis

        """
        try:
            analysis_result = {
                "symbol": symbol,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "success",
            }

            # Fetch stock data
            logger.info(f"Starting analysis for {symbol}")
            stock_data = self.data_fetcher.get_nse_stock_data(symbol, period)

            if stock_data.empty:
                return {
                    "symbol": symbol,
                    "status": "error",
                    "error": "Could not fetch stock data",
                }

            analysis_result["data_points"] = len(stock_data)
            analysis_result["current_price"] = float(stock_data["Close"].iloc[-1])

            # Get basic stock info
            if include_info:
                try:
                    stock_info = self.data_fetcher.get_stock_info(symbol)
                    if "error" not in stock_info:
                        analysis_result["stock_info"] = stock_info
                except Exception as e:
                    logger.warning(f"Could not fetch stock info for {symbol}: {str(e)}")

            # Technical analysis
            if include_technical:
                try:
                    # Calculate technical indicators
                    data_with_indicators = self.technical_analyzer.calculate_all_indicators(
                        stock_data
                    )

                    # Get latest indicators values
                    latest_indicators = self._extract_latest_indicators(data_with_indicators)
                    analysis_result["technical_indicators"] = latest_indicators

                    # Get trading signals
                    signals = self.technical_analyzer.get_latest_signals(data_with_indicators)
                    analysis_result["technical_signals"] = signals

                except Exception as e:
                    logger.error(f"Technical analysis failed for {symbol}: {str(e)}")
                    analysis_result["technical_error"] = str(e)

            # Price prediction
            if include_prediction:
                try:
                    # Prepare data for prediction
                    if include_technical:
                        # Use data with indicators if technical analysis was performed
                        prediction_data = data_with_indicators
                    else:
                        # Calculate indicators just for prediction
                        prediction_data = self.technical_analyzer.calculate_all_indicators(
                            stock_data
                        )

                    # Prepare features and train model
                    X, y = self.prediction_model.prepare_features(prediction_data)

                    if len(X) > 50:  # Ensure sufficient data
                        # Train model
                        metrics = self.prediction_model.train(X, y)
                        analysis_result["model_metrics"] = metrics

                        # Make prediction
                        prediction = self.prediction_model.predict_next_price(prediction_data)
                        analysis_result["prediction"] = prediction

                        # Get feature importance
                        feature_importance = self.prediction_model.get_feature_importance()
                        if feature_importance:
                            # Convert to list of dicts for JSON serialization
                            analysis_result["feature_importance"] = [
                                {"feature": feature, "importance": importance}
                                for feature, importance in list(feature_importance.items())[:10]
                            ]
                    else:
                        analysis_result["prediction_error"] = "Insufficient data for prediction"

                except Exception as e:
                    logger.error(f"Prediction failed for {symbol}: {str(e)}")
                    analysis_result["prediction_error"] = str(e)

            return analysis_result

        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "status": "error",
                "error": str(e),
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

    def _extract_latest_indicators(self, data_with_indicators: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract latest technical indicator values.

        Args:
            data_with_indicators: DataFrame with calculated indicators

        Returns:
            Dictionary with latest indicator values

        """
        try:
            latest = data_with_indicators.iloc[-1]

            indicators = {}

            # Moving averages
            for ma in ["MA_5", "MA_10", "MA_20", "MA_50"]:
                if ma in latest:
                    indicators[ma.lower()] = float(latest[ma]) if pd.notna(latest[ma]) else None

            # RSI
            if "RSI" in latest:
                indicators["rsi"] = float(latest["RSI"]) if pd.notna(latest["RSI"]) else None

            # MACD
            for macd_col in ["MACD", "MACD_Signal", "MACD_Histogram"]:
                if macd_col in latest:
                    key = macd_col.lower().replace("_", "_")
                    indicators[key] = (
                        float(latest[macd_col]) if pd.notna(latest[macd_col]) else None
                    )

            # Bollinger Bands
            for bb_col in ["BB_Upper", "BB_Lower", "BB_Middle"]:
                if bb_col in latest:
                    key = bb_col.lower().replace("_", "_")
                    indicators[key] = float(latest[bb_col]) if pd.notna(latest[bb_col]) else None

            # Other indicators
            other_indicators = [
                "ATR",
                "Stoch_K",
                "Williams_R",
                "Volume_Ratio",
                "Volatility",
            ]
            for indicator in other_indicators:
                if indicator in latest:
                    key = indicator.lower().replace("_", "_")
                    indicators[key] = (
                        float(latest[indicator]) if pd.notna(latest[indicator]) else None
                    )

            return indicators

        except Exception as e:
            logger.error(f"Error extracting indicators: {str(e)}")
            return {}

    def get_stock_info_only(self, symbol: str) -> Dict[str, Any]:
        """
        Get only basic stock information without analysis.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with stock information

        """
        try:
            stock_info = self.data_fetcher.get_stock_info(symbol)

            if "error" in stock_info:
                return {
                    "symbol": symbol,
                    "status": "error",
                    "error": stock_info["error"],
                }

            return {
                "symbol": symbol,
                "status": "success",
                "stock_info": stock_info,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as e:
            return {"symbol": symbol, "status": "error", "error": str(e)}

    def get_technical_analysis_only(self, symbol: str, period: str = "6mo") -> Dict[str, Any]:
        """
        Get only technical analysis without prediction.

        Args:
            symbol: Stock symbol
            period: Data period

        Returns:
            Dictionary with technical analysis

        """
        try:
            # Fetch data
            stock_data = self.data_fetcher.get_nse_stock_data(symbol, period)

            if stock_data.empty:
                return {
                    "symbol": symbol,
                    "status": "error",
                    "error": "Could not fetch stock data",
                }

            # Calculate indicators
            data_with_indicators = self.technical_analyzer.calculate_all_indicators(stock_data)

            # Get latest indicators
            latest_indicators = self._extract_latest_indicators(data_with_indicators)

            # Get signals
            signals = self.technical_analyzer.get_latest_signals(data_with_indicators)

            return {
                "symbol": symbol,
                "status": "success",
                "period": period,
                "current_price": float(stock_data["Close"].iloc[-1]),
                "technical_indicators": latest_indicators,
                "technical_signals": signals,
                "data_points": len(stock_data),
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {str(e)}")
            return {"symbol": symbol, "status": "error", "error": str(e)}

    def get_prediction_only(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """
        Get only price prediction without other analysis.

        Args:
            symbol: Stock symbol
            period: Data period for training

        Returns:
            Dictionary with prediction results

        """
        try:
            # Fetch data
            stock_data = self.data_fetcher.get_nse_stock_data(symbol, period)

            if stock_data.empty:
                return {
                    "symbol": symbol,
                    "status": "error",
                    "error": "Could not fetch stock data",
                }

            # Prepare data with indicators
            data_with_indicators = self.technical_analyzer.calculate_all_indicators(stock_data)

            # Prepare features and train model
            X, y = self.prediction_model.prepare_features(data_with_indicators)

            if len(X) < 50:
                return {
                    "symbol": symbol,
                    "status": "error",
                    "error": "Insufficient data for prediction",
                }

            # Train model
            metrics = self.prediction_model.train(X, y)

            # Make prediction
            prediction = self.prediction_model.predict_next_price(data_with_indicators)

            return {
                "symbol": symbol,
                "status": "success",
                "period": period,
                "current_price": float(stock_data["Close"].iloc[-1]),
                "prediction": prediction,
                "model_metrics": metrics,
                "model_type": self.model_type,
                "data_points": len(stock_data),
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {str(e)}")
            return {"symbol": symbol, "status": "error", "error": str(e)}

    def compare_stocks(
        self, symbols: List[str], period: str = "6mo", analysis_type: str = "prediction"
    ) -> Dict[str, Any]:
        """
        Compare multiple stocks.

        Args:
            symbols: List of stock symbols
            period: Data period
            analysis_type: Type of comparison (prediction, technical, info)

        Returns:
            Dictionary with comparison results

        """
        try:
            results = []

            for symbol in symbols:
                try:
                    if analysis_type == "prediction":
                        result = self.get_prediction_only(symbol, period)
                    elif analysis_type == "technical":
                        result = self.get_technical_analysis_only(symbol, period)
                    elif analysis_type == "info":
                        result = self.get_stock_info_only(symbol)
                    else:
                        result = self.analyze_stock(symbol, period)

                    if result["status"] == "success":
                        results.append(result)

                except Exception as e:
                    logger.warning(f"Failed to analyze {symbol}: {str(e)}")
                    continue

            if not results:
                return {
                    "status": "error",
                    "error": "No successful analyses for comparison",
                }

            # Generate comparison summary
            summary = self._generate_comparison_summary(results, analysis_type)

            return {
                "status": "success",
                "comparison_type": analysis_type,
                "symbols": [r["symbol"] for r in results],
                "results": results,
                "summary": summary,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as e:
            logger.error(f"Comparison failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _generate_comparison_summary(
        self, results: List[Dict], analysis_type: str
    ) -> Dict[str, Any]:
        """
        Generate summary for stock comparison.
        """
        try:
            if analysis_type == "prediction":
                predictions = [r.get("prediction", {}) for r in results if r.get("prediction")]
                if predictions:
                    changes = [p.get("percentage_change", 0) for p in predictions]
                    return {
                        "average_predicted_change": sum(changes) / len(changes),
                        "best_predicted_stock": max(
                            results,
                            key=lambda x: x.get("prediction", {}).get("percentage_change", 0),
                        )["symbol"],
                        "worst_predicted_stock": min(
                            results,
                            key=lambda x: x.get("prediction", {}).get("percentage_change", 0),
                        )["symbol"],
                        "positive_predictions": len([c for c in changes if c > 0]),
                        "negative_predictions": len([c for c in changes if c < 0]),
                    }

            elif analysis_type == "technical":
                signals = [
                    r.get("technical_signals", {}) for r in results if r.get("technical_signals")
                ]
                if signals:
                    overall_signals = [s.get("overall_signal", "Neutral") for s in signals]
                    return {
                        "bullish_count": overall_signals.count("Bullish"),
                        "bearish_count": overall_signals.count("Bearish"),
                        "neutral_count": overall_signals.count("Neutral"),
                        "market_sentiment": max(set(overall_signals), key=overall_signals.count),
                    }

            return {"message": "Summary not available for this analysis type"}

        except Exception as e:
            logger.error(f"Error generating comparison summary: {str(e)}")
            return {"error": "Could not generate summary"}
