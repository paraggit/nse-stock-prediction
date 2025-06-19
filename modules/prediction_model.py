"""
Simple Stock Price Prediction Model Module.
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class StockPredictionModel:
    """
    Simple machine learning model for stock price prediction.
    """

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False

        # Initialize components
        try:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
        except ImportError:
            logger.error("scikit-learn not available. Please install: pip install scikit-learn")
            self.scaler = None

    def _get_model(self):
        """
        Get the appropriate model based on model_type.
        """
        try:
            if self.model_type == "random_forest":
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            elif self.model_type == "gradient_boost":
                from sklearn.ensemble import GradientBoostingRegressor

                return GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=6)
            elif self.model_type == "linear_regression":
                from sklearn.linear_model import LinearRegression

                return LinearRegression()
            else:
                logger.warning(f"Unknown model type: {self.model_type}, using RandomForest")
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(n_estimators=50, random_state=42)
        except ImportError:
            logger.error("Required scikit-learn models not available")
            return None

    def prepare_features(
        self, data: pd.DataFrame, target_days: int = 1
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for machine learning.

        Args:
            data: DataFrame with stock data and technical indicators
            target_days: Number of days ahead to predict

        Returns:
            Tuple of (features, target)

        """
        try:
            df = data.copy()

            # Create target variable (future price)
            df["Target"] = df["Close"].shift(-target_days)

            # Define feature columns (use available columns)
            potential_features = [
                "Open",
                "High",
                "Low",
                "Volume",
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
                "Price_Change",
                "Price_Change_5",
                "Volume_Ratio",
                "Volatility",
                "HL_Spread",
            ]

            # Filter to only include available columns
            self.feature_columns = [col for col in potential_features if col in df.columns]

            if not self.feature_columns:
                # Fallback to basic features
                self.feature_columns = ["Open", "High", "Low", "Volume"]
                logger.warning("Using only basic OHLV features")

            # Remove rows with NaN values
            df = df.dropna()

            if df.empty:
                raise ValueError("No valid data after feature preparation")

            X = df[self.feature_columns]
            y = df["Target"]

            logger.info(
                f"Features prepared: {len(self.feature_columns)} features, {len(X)} samples"
            )
            return X, y

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the prediction model.

        Args:
            X: Features
            y: Target variable
            test_size: Proportion of data for testing

        Returns:
            Dictionary with training metrics

        """
        try:
            if self.scaler is None:
                raise ValueError("Scaler not available - scikit-learn not installed")

            # Get model
            self.model = self._get_model()
            if self.model is None:
                raise ValueError("Could not initialize model")

            # Split data (maintain temporal order)
            split_index = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Make predictions
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)

            # Calculate metrics
            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            self.is_trained = True

            metrics = {
                "train_r2": float(train_r2),
                "test_r2": float(test_r2),
                "test_mse": float(test_mse),
                "test_rmse": float(test_rmse),
                "test_mae": float(test_mae),
                "accuracy": float(test_r2 * 100),  # Convert R2 to percentage
            }

            logger.info(f"Model trained successfully. Test R2: {test_r2:.4f}")
            return metrics

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Features for prediction

        Returns:
            Array of predictions

        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def predict_next_price(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict the next day's closing price.

        Args:
            data: DataFrame with latest stock data and indicators

        Returns:
            Dictionary with prediction results

        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")

            # Get the latest row of data
            latest_data = data.iloc[-1:].copy()

            # Prepare features
            X_latest = latest_data[self.feature_columns]

            # Make prediction
            prediction = self.predict(X_latest)[0]
            current_price = data["Close"].iloc[-1]
            price_change = prediction - current_price
            percentage_change = (price_change / current_price) * 100

            # Simple confidence intervals (based on volatility)
            try:
                volatility = data["Close"].pct_change().std() * current_price
                confidence_upper = prediction + (1.96 * volatility)
                confidence_lower = prediction - (1.96 * volatility)
            except Exception as e:
                logger.warning(f"Error calculating confidence intervals: {str(e)}")
                confidence_upper = prediction * 1.05
                confidence_lower = prediction * 0.95

            result = {
                "current_price": float(current_price),
                "predicted_price": float(prediction),
                "price_change": float(price_change),
                "percentage_change": float(percentage_change),
                "confidence_upper": float(confidence_upper),
                "confidence_lower": float(confidence_lower),
                "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_type": self.model_type,
            }

            logger.info(f"Prediction made: {prediction:.2f} (Change: {percentage_change:.2f}%)")
            return result

        except Exception as e:
            logger.error(f"Error predicting next price: {str(e)}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance if available.

        Returns:
            Dictionary with feature importance scores

        """
        try:
            if not self.is_trained or not hasattr(self.model, "feature_importances_"):
                return {}

            importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))

            # Sort by importance
            sorted_importance = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            return sorted_importance

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}

    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model

        Returns:
            Boolean indicating success

        """
        try:
            if not self.is_trained:
                raise ValueError("Cannot save untrained model")

            import joblib

            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "model_type": self.model_type,
            }

            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True

        except ImportError:
            logger.error("joblib not available for model saving")
            return False
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to load the model from

        Returns:
            Boolean indicating success

        """
        try:
            import joblib

            model_data = joblib.load(filepath)

            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_columns = model_data["feature_columns"]
            self.model_type = model_data["model_type"]
            self.is_trained = True

            logger.info(f"Model loaded from {filepath}")
            return True

        except ImportError:
            logger.error("joblib not available for model loading")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
