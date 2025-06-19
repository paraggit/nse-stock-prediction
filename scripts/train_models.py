#!/usr/bin/env python3
"""
Model Training Script for NSE Stock Prediction Pipeline.

This script provides comprehensive model training capabilities including:
- Single stock model training
- Batch training for multiple stocks
- Sector-based training
- Model comparison and evaluation
- Hyperparameter optimization
- Model persistence and loading
- Training progress monitoring
- Performance analytics

Usage:
    poetry run train-models --help
    poetry run train-models train-single TCS
    poetry run train-models train-all --sector IT
    poetry run train-models compare-models RELIANCE
    poetry run train-models optimize-hyperparams TCS

"""

import asyncio
import json
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import typer
from loguru import logger
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from modules.prediction_model import StockPredictionModel
from modules.stock_analyzer import StockAnalyzer
from modules.stock_data_fetcher import StockDataFetcher
from modules.technical_indicators import TechnicalIndicators

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Initialize Typer app and console
app = typer.Typer(
    name="train-models",
    help="🤖 NSE Stock Prediction Model Training Suite",
    rich_markup_mode="rich",
)
console = Console()

# Stock configurations
POPULAR_STOCKS = [
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
    "ADANIPORTS",
    "BHARTIARTL",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "GRASIM",
    "HDFCBANK",
    "HEROMOTOCO",
    "HINDALCO",
    "INDUSINDBK",
    "JSWSTEEL",
    "M&M",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "SHREECEM",
    "TATAMOTORS",
    "TATASTEEL",
    "TECHM",
    "UPL",
]

SECTOR_STOCKS = {
    "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "MINDTREE", "MPHASIS", "LTI"],
    "BANKING": [
        "HDFC",
        "ICICIBANK",
        "SBIN",
        "KOTAKBANK",
        "AXISBANK",
        "INDUSINDBK",
        "FEDERALBNK",
        "PNB",
    ],
    "FMCG": [
        "HINDUNILVR",
        "ITC",
        "NESTLEIND",
        "BRITANNIA",
        "DABUR",
        "GODREJCP",
        "MARICO",
        "COLPAL",
    ],
    "AUTO": [
        "MARUTI",
        "TATAMOTORS",
        "BAJAJ-AUTO",
        "M&M",
        "HEROMOTOCO",
        "EICHERMOT",
        "ASHOKLEY",
        "BAJAJFINSV",
    ],
    "PHARMA": [
        "SUNPHARMA",
        "DRREDDY",
        "CIPLA",
        "LUPIN",
        "BIOCON",
        "AUROPHARMA",
        "CADILAHC",
        "GLENMARK",
    ],
    "ENERGY": [
        "RELIANCE",
        "ONGC",
        "BPCL",
        "IOC",
        "HINDPETRO",
        "GAIL",
        "ADANIGREEN",
        "NTPC",
    ],
    "METALS": [
        "TATASTEEL",
        "HINDALCO",
        "VEDL",
        "JSWSTEEL",
        "COALINDIA",
        "NMDC",
        "SAIL",
        "MOIL",
    ],
    "TELECOM": ["BHARTIARTL", "IDEA", "RCOM"],
    "CEMENT": ["ULTRACEMCO", "SHREECEM", "ACC", "AMBUJACEMENT", "JKCEMENT"],
    "REALTY": ["DLF", "GODREJPROP", "BRIGADE", "PRESTIGE", "SOBHA"],
}

MODEL_TYPES = ["random_forest", "gradient_boost", "linear_regression"]
PERIODS = ["1y", "2y", "3y", "5y"]


class TrainingConfig:
    """
    Configuration for model training.
    """

    def __init__(self):
        self.models_dir = Path("models/saved_models")
        self.results_dir = Path("training_results")
        self.logs_dir = Path("logs/training")

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Training parameters
        self.default_period = "2y"
        self.default_model = "random_forest"
        self.test_size = 0.2
        self.cv_folds = 5
        self.n_jobs = -1
        self.random_state = 42

        # Performance thresholds
        self.min_accuracy = 60.0  # Minimum acceptable accuracy
        self.min_r2_score = 0.5  # Minimum R² score
        self.max_training_time = 300  # Maximum training time per stock (seconds)


config = TrainingConfig()


class ModelTrainer:
    """
    Advanced model trainer with monitoring and optimization.
    """

    def __init__(self):
        self.training_stats = {
            "total_trained": 0,
            "successful": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None,
        }
        self.results = []

    def train_single_stock(
        self,
        symbol: str,
        model_type: str = "random_forest",
        period: str = "2y",
        save_model: bool = True,
        optimize_hyperparams: bool = False,
        cross_validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Train model for a single stock with comprehensive evaluation.
        """

        start_time = time.time()

        try:
            console.print(f"🤖 Training {model_type} model for [bold cyan]{symbol}[/bold cyan]...")

            # Initialize components
            analyzer = StockAnalyzer(model_type=model_type)

            # Fetch and prepare data
            console.print(f"📊 Fetching data for {symbol} (period: {period})")
            data = analyzer.data_fetcher.get_nse_stock_data(symbol, period)

            if data.empty:
                raise ValueError(f"No data available for {symbol}")

            console.print(f"✅ Fetched {len(data)} data points")

            # Add technical indicators
            data_with_indicators = analyzer.technical_analyzer.calculate_all_indicators(data)

            # Prepare features
            X, y = analyzer.prediction_model.prepare_features(data_with_indicators)
            console.print(f"🔧 Prepared {len(analyzer.prediction_model.feature_columns)} features")

            # Hyperparameter optimization
            if optimize_hyperparams:
                console.print("🎯 Optimizing hyperparameters...")
                opt_results = analyzer.prediction_model.optimize_hyperparameters(X, y)
                console.print(f"✅ Best parameters: {opt_results.get('best_params', {})}")

            # Train model
            console.print("🚀 Training model...")
            metrics = analyzer.prediction_model.train(X, y)

            training_time = time.time() - start_time

            # Cross-validation
            cv_score = None
            if cross_validate:
                console.print("🔄 Performing cross-validation...")
                cv_score = self._cross_validate_model(analyzer.prediction_model, X, y)
                console.print(f"📈 CV Score: {cv_score:.3f}")

            # Feature importance
            feature_importance = analyzer.prediction_model.get_feature_importance()
            top_features = list(feature_importance.keys())[:10] if feature_importance else []

            # Model evaluation
            evaluation = self._evaluate_model_performance(metrics, training_time)

            # Save model
            model_path = None
            if save_model and evaluation["meets_threshold"]:
                model_filename = (
                    f"{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                )
                model_path = config.models_dir / model_filename

                success = analyzer.prediction_model.save_model(str(model_path))
                if success:
                    console.print(f"💾 Model saved: {model_path}")
                else:
                    console.print("⚠️ Failed to save model", style="yellow")

            # Prepare result
            result = {
                "symbol": symbol,
                "status": "success",
                "model_type": model_type,
                "period": period,
                "training_time": round(training_time, 2),
                "data_points": len(data),
                "feature_count": len(analyzer.prediction_model.feature_columns),
                "metrics": {
                    "accuracy": round(metrics["accuracy"], 2),
                    "r2_score": round(metrics["test_r2"], 3),
                    "rmse": round(metrics["test_rmse"], 3),
                    "mae": round(metrics["test_mae"], 3),
                    "train_r2": round(metrics["train_r2"], 3),
                    "cv_score": round(cv_score, 3) if cv_score else None,
                },
                "evaluation": evaluation,
                "top_features": top_features[:5],
                "model_path": str(model_path) if model_path else None,
                "timestamp": datetime.now().isoformat(),
            }

            # Log result
            if evaluation["meets_threshold"]:
                console.print(
                    f"✅ Training successful - Accuracy: {metrics['accuracy']:.1f}%",
                    style="green",
                )
            else:
                console.print(
                    f"⚠️ Training completed but below threshold - Accuracy: {metrics['accuracy']:.1f}%",
                    style="yellow",
                )

            return result

        except Exception as e:
            training_time = time.time() - start_time
            error_msg = str(e)

            console.print(f"❌ Training failed for {symbol}: {error_msg}", style="red")

            return {
                "symbol": symbol,
                "status": "failed",
                "model_type": model_type,
                "period": period,
                "training_time": round(training_time, 2),
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
            }

    def _cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Perform cross-validation on the model.
        """
        from sklearn.model_selection import cross_val_score

        try:
            # Scale features for cross-validation
            X_scaled = model.scaler.fit_transform(X)

            # Perform cross-validation
            cv_scores = cross_val_score(
                model.model,
                X_scaled,
                y,
                cv=config.cv_folds,
                scoring="r2",
                n_jobs=config.n_jobs,
            )

            return cv_scores.mean()
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
            return 0.0

    def _evaluate_model_performance(
        self, metrics: Dict[str, float], training_time: float
    ) -> Dict[str, Any]:
        """
        Evaluate if model meets performance thresholds.
        """

        accuracy = metrics.get("accuracy", 0)
        r2_score = metrics.get("test_r2", 0)

        meets_accuracy = accuracy >= config.min_accuracy
        meets_r2 = r2_score >= config.min_r2_score
        within_time_limit = training_time <= config.max_training_time

        meets_threshold = meets_accuracy and meets_r2 and within_time_limit

        return {
            "meets_threshold": meets_threshold,
            "meets_accuracy": meets_accuracy,
            "meets_r2": meets_r2,
            "within_time_limit": within_time_limit,
            "performance_grade": self._get_performance_grade(accuracy, r2_score),
        }

    def _get_performance_grade(self, accuracy: float, r2_score: float) -> str:
        """
        Get performance grade based on metrics.
        """

        if accuracy >= 85 and r2_score >= 0.8:
            return "A+"
        elif accuracy >= 80 and r2_score >= 0.75:
            return "A"
        elif accuracy >= 75 and r2_score >= 0.7:
            return "B+"
        elif accuracy >= 70 and r2_score >= 0.65:
            return "B"
        elif accuracy >= 65 and r2_score >= 0.6:
            return "C+"
        elif accuracy >= 60 and r2_score >= 0.5:
            return "C"
        else:
            return "D"


# Training commands
@app.command()
def train_single(
    symbol: str = typer.Argument(..., help="Stock symbol to train"),
    model_type: str = typer.Option("random_forest", help="Model type"),
    period: str = typer.Option("2y", help="Data period"),
    save_model: bool = typer.Option(True, help="Save trained model"),
    optimize: bool = typer.Option(False, help="Optimize hyperparameters"),
    cross_validate: bool = typer.Option(True, help="Perform cross-validation"),
    output_file: Optional[str] = typer.Option(None, help="Save results to JSON file"),
) -> None:
    """
    🎯 Train model for a single stock with detailed evaluation.
    """

    console.print("\n🤖 [bold blue]Single Stock Model Training[/bold blue]\n")
    console.print(f"📊 Stock: [bold cyan]{symbol.upper()}[/bold cyan]")
    console.print(f"🔧 Model: [bold green]{model_type}[/bold green]")
    console.print(f"📅 Period: [bold yellow]{period}[/bold yellow]")
    console.print()

    trainer = ModelTrainer()

    with console.status("[bold green]Training model..."):
        result = trainer.train_single_stock(
            symbol=symbol.upper(),
            model_type=model_type,
            period=period,
            save_model=save_model,
            optimize_hyperparams=optimize,
            cross_validate=cross_validate,
        )

    # Display results
    _display_single_training_result(result)

    # Save results
    if output_file:
        _save_training_results([result], output_file)
        console.print(f"\n💾 Results saved to: [bold green]{output_file}[/bold green]")


@app.command()
def train_all(
    stocks: Optional[List[str]] = typer.Option(None, help="List of stock symbols"),
    model_type: str = typer.Option("random_forest", help="Model type"),
    period: str = typer.Option("2y", help="Data period"),
    sector: Optional[str] = typer.Option(None, help="Train stocks from specific sector"),
    limit: int = typer.Option(20, help="Maximum number of stocks to train"),
    save_models: bool = typer.Option(True, help="Save trained models"),
    parallel: bool = typer.Option(True, help="Train models in parallel"),
    optimize: bool = typer.Option(False, help="Optimize hyperparameters"),
    output_file: Optional[str] = typer.Option(None, help="Save results to JSON file"),
) -> None:
    """
    🚀 Train models for multiple stocks with progress monitoring.
    """

    # Determine stock list
    if sector and sector.upper() in SECTOR_STOCKS:
        stock_list = SECTOR_STOCKS[sector.upper()]
        console.print(f"📊 Training models for [bold cyan]{sector.upper()}[/bold cyan] sector")
    elif stocks:
        stock_list = [s.upper() for s in stocks]
        console.print("📊 Training models for provided stock list")
    else:
        stock_list = POPULAR_STOCKS[:limit]
        console.print(f"📊 Training models for top {limit} popular stocks")

    stock_list = stock_list[:limit]  # Apply limit

    console.print(f"🔧 Model: [bold green]{model_type}[/bold green]")
    console.print(f"📅 Period: [bold yellow]{period}[/bold yellow]")
    console.print(f"📈 Stocks ({len(stock_list)}): {', '.join(stock_list[:10])}")
    if len(stock_list) > 10:
        console.print(f"    ... and {len(stock_list) - 10} more")
    console.print()

    trainer = ModelTrainer()
    results = []

    if parallel and len(stock_list) > 1:
        # Parallel training
        results = _train_parallel(trainer, stock_list, model_type, period, save_models, optimize)
    else:
        # Sequential training with progress bar
        results = _train_sequential(trainer, stock_list, model_type, period, save_models, optimize)

    # Display summary
    _display_training_summary(results, model_type)

    # Save results
    if output_file:
        _save_training_results(results, output_file)
        console.print(f"\n💾 Results saved to: [bold green]{output_file}[/bold green]")


@app.command()
def compare_models(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    models: List[str] = typer.Option(MODEL_TYPES, help="Models to compare"),
    period: str = typer.Option("2y", help="Data period"),
    save_best: bool = typer.Option(True, help="Save the best performing model"),
    output_file: Optional[str] = typer.Option(None, help="Save comparison to JSON file"),
) -> None:
    """
    ⚖️ Compare different models for a single stock.
    """

    console.print(f"\n⚖️ [bold blue]Model Comparison for {symbol.upper()}[/bold blue]\n")
    console.print(f"📊 Models to compare: {', '.join(models)}")
    console.print(f"📅 Period: [bold yellow]{period}[/bold yellow]")
    console.print()

    trainer = ModelTrainer()
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Comparing models...", total=len(models))

        for model_type in models:
            progress.update(task, description=f"Training {model_type}...")

            result = trainer.train_single_stock(
                symbol=symbol.upper(),
                model_type=model_type,
                period=period,
                save_model=False,  # Don't save during comparison
                optimize_hyperparams=False,
                cross_validate=True,
            )

            results.append(result)
            progress.advance(task)

    # Display comparison
    _display_model_comparison(results, symbol.upper())

    # Save best model
    if save_best:
        best_result = _get_best_model_result(results)
        if best_result and best_result["status"] == "success":
            console.print(
                f"\n🏆 Retraining and saving best model: [bold green]{best_result['model_type']}[/bold green]"
            )

            # Retrain and save the best model
            final_result = trainer.train_single_stock(
                symbol=symbol.upper(),
                model_type=best_result["model_type"],
                period=period,
                save_model=True,
                optimize_hyperparams=True,
                cross_validate=False,
            )

            if final_result["status"] == "success":
                console.print(
                    f"✅ Best model saved: [green]{final_result.get('model_path', 'Unknown path')}[/green]"
                )

    # Save comparison results
    if output_file:
        comparison_data = {
            "symbol": symbol.upper(),
            "comparison_date": datetime.now().isoformat(),
            "period": period,
            "results": results,
            "best_model": _get_best_model_result(results),
            "summary": _generate_comparison_summary(results),
        }
        _save_training_results([comparison_data], output_file)
        console.print(f"\n💾 Comparison saved to: [bold green]{output_file}[/bold green]")


@app.command()
def optimize_hyperparams(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    model_type: str = typer.Option("random_forest", help="Model type to optimize"),
    period: str = typer.Option("2y", help="Data period"),
    save_model: bool = typer.Option(True, help="Save optimized model"),
    output_file: Optional[str] = typer.Option(None, help="Save results to JSON file"),
) -> None:
    """
    🎯 Optimize hyperparameters for a specific model and stock.
    """

    console.print("\n🎯 [bold blue]Hyperparameter Optimization[/bold blue]\n")
    console.print(f"📊 Stock: [bold cyan]{symbol.upper()}[/bold cyan]")
    console.print(f"🔧 Model: [bold green]{model_type}[/bold green]")
    console.print(f"📅 Period: [bold yellow]{period}[/bold yellow]")
    console.print()

    trainer = ModelTrainer()

    with console.status("[bold green]Optimizing hyperparameters..."):
        result = trainer.train_single_stock(
            symbol=symbol.upper(),
            model_type=model_type,
            period=period,
            save_model=save_model,
            optimize_hyperparams=True,
            cross_validate=True,
        )

    # Display results
    _display_single_training_result(result)

    if output_file:
        _save_training_results([result], output_file)
        console.print(f"\n💾 Results saved to: [bold green]{output_file}[/bold green]")


@app.command()
def train_sector(
    sector: str = typer.Argument(..., help="Sector name (IT, BANKING, FMCG, etc.)"),
    model_type: str = typer.Option("random_forest", help="Model type"),
    period: str = typer.Option("2y", help="Data period"),
    save_models: bool = typer.Option(True, help="Save trained models"),
    parallel: bool = typer.Option(True, help="Train in parallel"),
    output_file: Optional[str] = typer.Option(None, help="Save results to JSON file"),
) -> None:
    """
    🏭 Train models for all stocks in a specific sector.
    """

    sector = sector.upper()

    if sector not in SECTOR_STOCKS:
        console.print(f"❌ Unknown sector: {sector}", style="red")
        console.print(f"Available sectors: {', '.join(SECTOR_STOCKS.keys())}")
        raise typer.Exit(1)

    stock_list = SECTOR_STOCKS[sector]

    console.print(f"\n🏭 [bold blue]Sector Training: {sector}[/bold blue]\n")
    console.print(f"📊 Stocks ({len(stock_list)}): {', '.join(stock_list)}")
    console.print(f"🔧 Model: [bold green]{model_type}[/bold green]")
    console.print(f"📅 Period: [bold yellow]{period}[/bold yellow]")
    console.print()

    trainer = ModelTrainer()

    if parallel and len(stock_list) > 1:
        results = _train_parallel(trainer, stock_list, model_type, period, save_models, False)
    else:
        results = _train_sequential(trainer, stock_list, model_type, period, save_models, False)

    # Display sector-specific summary
    _display_sector_summary(results, sector, model_type)

    if output_file:
        sector_data = {
            "sector": sector,
            "training_date": datetime.now().isoformat(),
            "model_type": model_type,
            "period": period,
            "results": results,
            "summary": _generate_sector_summary(results),
        }
        _save_training_results([sector_data], output_file)
        console.print(f"\n💾 Sector results saved to: [bold green]{output_file}[/bold green]")


@app.command()
def list_models(
    symbol: Optional[str] = typer.Option(None, help="Filter by stock symbol"),
    model_type: Optional[str] = typer.Option(None, help="Filter by model type"),
    limit: int = typer.Option(50, help="Maximum number of models to display"),
) -> None:
    """
    📂 List all saved models with details.
    """

    console.print("\n📂 [bold blue]Saved Models[/bold blue]\n")

    models_dir = config.models_dir

    if not models_dir.exists():
        console.print("❌ No models directory found", style="red")
        return

    # Get all model files
    model_files = list(models_dir.glob("*.joblib"))

    if not model_files:
        console.print("📭 No saved models found", style="yellow")
        return

    # Parse model information
    models_info = []

    for model_file in model_files:
        try:
            # Parse filename: symbol_modeltype_timestamp.joblib
            parts = model_file.stem.split("_")
            if len(parts) >= 2:
                file_symbol = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
                file_model_type = parts[-2] if len(parts) > 2 else parts[1]

                # Apply filters
                if symbol and file_symbol.upper() != symbol.upper():
                    continue
                if model_type and file_model_type != model_type:
                    continue

                # Get file stats
                file_stat = model_file.stat()
                file_size = file_stat.st_size / (1024 * 1024)  # MB
                created_date = datetime.fromtimestamp(file_stat.st_ctime)

                models_info.append(
                    {
                        "symbol": file_symbol.upper(),
                        "model_type": file_model_type,
                        "file_path": str(model_file),
                        "file_size_mb": round(file_size, 2),
                        "created_date": created_date,
                        "file_name": model_file.name,
                    }
                )
        except Exception as e:
            logger.warning(f"Error parsing model file {model_file}: {e}")
            continue

    if not models_info:
        console.print("📭 No models found matching the criteria", style="yellow")
        return

    # Sort by creation date (newest first)
    models_info.sort(key=lambda x: x["created_date"], reverse=True)
    models_info = models_info[:limit]

    # Display table
    table = Table(title=f"Saved Models ({len(models_info)} found)")
    table.add_column("Symbol", style="cyan")
    table.add_column("Model Type", style="green")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Created", style="yellow")
    table.add_column("File Name", style="dim")

    for model in models_info:
        table.add_row(
            model["symbol"],
            model["model_type"],
            f"{model['file_size_mb']:.2f}",
            model["created_date"].strftime("%Y-%m-%d %H:%M"),
            model["file_name"],
        )

    console.print(table)
    console.print(f"\n📊 Total models: {len(models_info)} (showing {min(limit, len(models_info))})")


# Helper functions for training operations


def _train_parallel(
    trainer: ModelTrainer,
    stock_list: List[str],
    model_type: str,
    period: str,
    save_models: bool,
    optimize: bool,
) -> List[Dict[str, Any]]:
    """
    Train models in parallel using ThreadPoolExecutor.
    """

    console.print("🔄 Training models in parallel...")

    results = []
    max_workers = min(4, len(stock_list))  # Limit concurrent training

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training models...", total=len(stock_list))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all training tasks
            future_to_symbol = {
                executor.submit(
                    trainer.train_single_stock,
                    symbol,
                    model_type,
                    period,
                    save_models,
                    optimize,
                    False,  # No cross-validation in parallel mode for speed
                ): symbol
                for symbol in stock_list
            }

            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Update progress
                    progress.update(task, advance=1, description=f"Completed {symbol}")

                    # Log immediate result
                    if result["status"] == "success":
                        accuracy = result["metrics"]["accuracy"]
                        console.print(f"✅ {symbol}: {accuracy:.1f}% accuracy", style="green")
                    else:
                        console.print(f"❌ {symbol}: {result.get('error', 'Failed')}", style="red")

                except Exception as e:
                    console.print(f"❌ {symbol}: Exception - {str(e)}", style="red")
                    results.append(
                        {
                            "symbol": symbol,
                            "status": "failed",
                            "error": f"Exception: {str(e)}",
                            "model_type": model_type,
                            "period": period,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    progress.advance(task)

    return results


def _train_sequential(
    trainer: ModelTrainer,
    stock_list: List[str],
    model_type: str,
    period: str,
    save_models: bool,
    optimize: bool,
) -> List[Dict[str, Any]]:
    """
    Train models sequentially with detailed progress.
    """

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training models...", total=len(stock_list))

        for symbol in stock_list:
            progress.update(task, description=f"Training {symbol}...")

            result = trainer.train_single_stock(
                symbol=symbol,
                model_type=model_type,
                period=period,
                save_model=save_models,
                optimize_hyperparams=optimize,
                cross_validate=True,
            )

            results.append(result)
            progress.advance(task)

            # Brief status update
            if result["status"] == "success":
                accuracy = result["metrics"]["accuracy"]
                grade = result["evaluation"]["performance_grade"]
                console.print(f"✅ {symbol}: {accuracy:.1f}% (Grade: {grade})", style="green")
            else:
                console.print(f"❌ {symbol}: {result.get('error', 'Failed')}", style="red")

    return results


def _display_single_training_result(result: Dict[str, Any]) -> None:
    """
    Display detailed results for single stock training.
    """

    if result["status"] == "success":
        metrics = result["metrics"]
        evaluation = result["evaluation"]

        # Create result panel
        result_text = f"""
[bold green]✅ Training Successful[/bold green]

📊 [bold]Performance Metrics:[/bold]
   • Accuracy: [bold cyan]{metrics['accuracy']}%[/bold cyan]
   • R² Score: [bold cyan]{metrics['r2_score']:.3f}[/bold cyan]
   • RMSE: [bold cyan]{metrics['rmse']:.3f}[/bold cyan]
   • MAE: [bold cyan]{metrics['mae']:.3f}[/bold cyan]
   • Grade: [bold yellow]{evaluation['performance_grade']}[/bold yellow]

🔧 [bold]Training Details:[/bold]
   • Training Time: [cyan]{result['training_time']}s[/cyan]
   • Data Points: [cyan]{result['data_points']}[/cyan]
   • Features Used: [cyan]{result['feature_count']}[/cyan]

🏆 [bold]Top Features:[/bold]
   {', '.join(result.get('top_features', []))}
        """

        if result.get("model_path"):
            result_text += f"\n💾 [bold]Model Saved:[/bold] [green]{result['model_path']}[/green]"

        panel = Panel(
            result_text.strip(),
            title=f"[bold]{result['symbol']} - {result['model_type']}[/bold]",
        )
        console.print(panel)

    else:
        error_text = f"""
[bold red]❌ Training Failed[/bold red]

[bold]Error:[/bold] [red]{result.get('error', 'Unknown error')}[/red]
[bold]Training Time:[/bold] [cyan]{result['training_time']}s[/cyan]
        """

        panel = Panel(
            error_text.strip(),
            title=f"[bold]{result['symbol']} - {result['model_type']}[/bold]",
        )
        console.print(panel)


def _display_training_summary(results: List[Dict[str, Any]], model_type: str) -> None:
    """
    Display comprehensive training summary.
    """

    successful_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] == "failed"]

    total_count = len(results)
    success_count = len(successful_results)
    failure_count = len(failed_results)
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0

    # Calculate statistics
    if successful_results:
        avg_accuracy = sum(r["metrics"]["accuracy"] for r in successful_results) / len(
            successful_results
        )
        avg_r2 = sum(r["metrics"]["r2_score"] for r in successful_results) / len(successful_results)
        avg_training_time = sum(r["training_time"] for r in successful_results) / len(
            successful_results
        )

        best_result = max(successful_results, key=lambda x: x["metrics"]["accuracy"])
        worst_result = min(successful_results, key=lambda x: x["metrics"]["accuracy"])

        # Grade distribution
        grades = [r["evaluation"]["performance_grade"] for r in successful_results]
        grade_counts = {grade: grades.count(grade) for grade in set(grades)}
    else:
        avg_accuracy = avg_r2 = avg_training_time = 0
        best_result = worst_result = None
        grade_counts = {}

    # Summary panel
    summary_text = f"""
📊 [bold]Overall Statistics:[/bold]
   • Total Stocks: [cyan]{total_count}[/cyan]
   • Successful: [green]{success_count}[/green] ({success_rate:.1f}%)
   • Failed: [red]{failure_count}[/red] ({100-success_rate:.1f}%)
   • Model Type: [yellow]{model_type}[/yellow]
    """

    if successful_results:
        summary_text += f"""
📈 [bold]Performance Metrics:[/bold]
   • Average Accuracy: [cyan]{avg_accuracy:.1f}%[/cyan]
   • Average R² Score: [cyan]{avg_r2:.3f}[/cyan]
   • Average Training Time: [cyan]{avg_training_time:.1f}s[/cyan]

🏆 [bold]Best Performer:[/bold]
   • Stock: [green]{best_result['symbol']}[/green]
   • Accuracy: [green]{best_result['metrics']['accuracy']:.1f}%[/green]
   • Grade: [green]{best_result['evaluation']['performance_grade']}[/green]

📉 [bold]Needs Improvement:[/bold]
   • Stock: [yellow]{worst_result['symbol']}[/yellow]
   • Accuracy: [yellow]{worst_result['metrics']['accuracy']:.1f}%[/yellow]
   • Grade: [yellow]{worst_result['evaluation']['performance_grade']}[/yellow]

📋 [bold]Grade Distribution:[/bold]
        """

        for grade, count in sorted(grade_counts.items()):
            summary_text += f"   • Grade {grade}: [cyan]{count}[/cyan] stocks\n"

    panel = Panel(summary_text.strip(), title="[bold blue]Training Summary[/bold blue]")
    console.print(panel)

    # Detailed results table
    if successful_results:
        console.print("\n📊 [bold]Detailed Results:[/bold]")

        table = Table()
        table.add_column("Symbol", style="cyan")
        table.add_column("Accuracy", justify="right")
        table.add_column("R² Score", justify="right")
        table.add_column("Grade", style="yellow")
        table.add_column("Time (s)", justify="right")
        table.add_column("Status", style="green")

        # Sort by accuracy (descending)
        sorted_results = sorted(
            successful_results, key=lambda x: x["metrics"]["accuracy"], reverse=True
        )

        for result in sorted_results[:20]:  # Show top 20
            metrics = result["metrics"]
            evaluation = result["evaluation"]

            table.add_row(
                result["symbol"],
                f"{metrics['accuracy']:.1f}%",
                f"{metrics['r2_score']:.3f}",
                evaluation["performance_grade"],
                f"{result['training_time']:.1f}",
                "✅ Success",
            )

        console.print(table)

        if len(successful_results) > 20:
            console.print(f"... and {len(successful_results) - 20} more successful trainings")

    # Failed results
    if failed_results:
        console.print(f"\n❌ [bold red]Failed Trainings ({len(failed_results)}):[/bold red]")

        fail_table = Table()
        fail_table.add_column("Symbol", style="red")
        fail_table.add_column("Error", style="yellow")
        fail_table.add_column("Time (s)", justify="right")

        for result in failed_results[:10]:  # Show first 10 failures
            fail_table.add_row(
                result["symbol"],
                result.get("error", "Unknown error")[:50] + "..."
                if len(result.get("error", "")) > 50
                else result.get("error", "Unknown error"),
                f"{result['training_time']:.1f}",
            )

        console.print(fail_table)

        if len(failed_results) > 10:
            console.print(f"... and {len(failed_results) - 10} more failures")


def _display_model_comparison(results: List[Dict[str, Any]], symbol: str) -> None:
    """
    Display model comparison results.
    """

    successful_results = [r for r in results if r["status"] == "success"]

    if not successful_results:
        console.print("❌ No successful model training to compare", style="red")
        return

    # Comparison table
    table = Table(title=f"Model Comparison for {symbol}")
    table.add_column("Model Type", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("R² Score", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("CV Score", justify="right")
    table.add_column("Grade", style="yellow")
    table.add_column("Time (s)", justify="right")

    # Sort by accuracy
    sorted_results = sorted(
        successful_results, key=lambda x: x["metrics"]["accuracy"], reverse=True
    )

    for i, result in enumerate(sorted_results):
        metrics = result["metrics"]
        evaluation = result["evaluation"]

        # Highlight best model
        style = "bold green" if i == 0 else None

        table.add_row(
            result["model_type"],
            f"{metrics['accuracy']:.1f}%",
            f"{metrics['r2_score']:.3f}",
            f"{metrics['rmse']:.3f}",
            f"{metrics.get('cv_score', 0):.3f}" if metrics.get("cv_score") else "N/A",
            evaluation["performance_grade"],
            f"{result['training_time']:.1f}",
            style=style,
        )

    console.print(table)

    # Best model summary
    best_result = sorted_results[0]
    console.print(f"\n🏆 [bold green]Best Model: {best_result['model_type']}[/bold green]")
    console.print(f"   • Accuracy: [cyan]{best_result['metrics']['accuracy']:.1f}%[/cyan]")
    console.print(f"   • R² Score: [cyan]{best_result['metrics']['r2_score']:.3f}[/cyan]")
    console.print(f"   • Grade: [yellow]{best_result['evaluation']['performance_grade']}[/yellow]")


def _display_sector_summary(results: List[Dict[str, Any]], sector: str, model_type: str) -> None:
    """
    Display sector-specific training summary.
    """

    successful_results = [r for r in results if r["status"] == "success"]

    console.print(f"\n🏭 [bold blue]Sector Summary: {sector}[/bold blue]")
    console.print(f"🔧 Model: [bold green]{model_type}[/bold green]")

    if successful_results:
        avg_accuracy = sum(r["metrics"]["accuracy"] for r in successful_results) / len(
            successful_results
        )
        sector_grade = _get_sector_grade(avg_accuracy)

        console.print(
            f"📊 Sector Performance: [bold yellow]{sector_grade}[/bold yellow] (Avg: {avg_accuracy:.1f}%)"
        )

        # Top performers in sector
        top_performers = sorted(
            successful_results, key=lambda x: x["metrics"]["accuracy"], reverse=True
        )[:3]

        console.print(f"\n🏆 [bold]Top Performers in {sector}:[/bold]")
        for i, result in enumerate(top_performers, 1):
            console.print(
                f"   {i}. [cyan]{result['symbol']}[/cyan]: {result['metrics']['accuracy']:.1f}% (Grade: {result['evaluation']['performance_grade']})"
            )

    _display_training_summary(results, model_type)


def _get_best_model_result(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Get the best performing model from comparison results.
    """

    successful_results = [r for r in results if r["status"] == "success"]

    if not successful_results:
        return None

    # Best model based on accuracy
    return max(successful_results, key=lambda x: x["metrics"]["accuracy"])


def _generate_comparison_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics for model comparison.
    """

    successful_results = [r for r in results if r["status"] == "success"]

    if not successful_results:
        return {"error": "No successful results to summarize"}

    accuracies = [r["metrics"]["accuracy"] for r in successful_results]
    [r["metrics"]["r2_score"] for r in successful_results]

    return {
        "best_accuracy": max(accuracies),
        "worst_accuracy": min(accuracies),
        "avg_accuracy": sum(accuracies) / len(accuracies),
        "accuracy_range": max(accuracies) - min(accuracies),
        "best_model": max(successful_results, key=lambda x: x["metrics"]["accuracy"])["model_type"],
        "models_compared": len(successful_results),
        "performance_variance": np.var(accuracies) if len(accuracies) > 1 else 0,
    }


def _generate_sector_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics for sector training.
    """

    successful_results = [r for r in results if r["status"] == "success"]
    total_results = len(results)

    if not successful_results:
        return {
            "total_stocks": total_results,
            "successful_trainings": 0,
            "success_rate": 0,
            "avg_accuracy": 0,
            "sector_grade": "F",
        }

    accuracies = [r["metrics"]["accuracy"] for r in successful_results]
    avg_accuracy = sum(accuracies) / len(accuracies)

    return {
        "total_stocks": total_results,
        "successful_trainings": len(successful_results),
        "success_rate": (len(successful_results) / total_results) * 100,
        "avg_accuracy": avg_accuracy,
        "sector_grade": _get_sector_grade(avg_accuracy),
        "best_performer": max(successful_results, key=lambda x: x["metrics"]["accuracy"])["symbol"],
        "worst_performer": min(successful_results, key=lambda x: x["metrics"]["accuracy"])[
            "symbol"
        ],
    }


def _get_sector_grade(avg_accuracy: float) -> str:
    """
    Get sector grade based on average accuracy.
    """

    if avg_accuracy >= 85:
        return "A+"
    elif avg_accuracy >= 80:
        return "A"
    elif avg_accuracy >= 75:
        return "B+"
    elif avg_accuracy >= 70:
        return "B"
    elif avg_accuracy >= 65:
        return "C+"
    elif avg_accuracy >= 60:
        return "C"
    else:
        return "D"


def _save_training_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save training results to JSON file.
    """

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for JSON serialization
    json_data = {
        "export_timestamp": datetime.now().isoformat(),
        "export_version": "1.0",
        "results": results,
        "summary": {
            "total_results": len(results),
            "successful_results": len([r for r in results if r.get("status") == "success"]),
            "failed_results": len([r for r in results if r.get("status") == "failed"]),
        },
    }

    try:
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save results to {output_file}: {e}")
        console.print(f"❌ Failed to save results: {e}", style="red")


# Additional utility commands
@app.command()
def cleanup_models(
    older_than_days: int = typer.Option(30, help="Remove models older than N days"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm deletion without prompt"),
) -> None:
    """
    🧹 Clean up old model files.
    """

    console.print("\n🧹 [bold blue]Model Cleanup[/bold blue]")
    console.print(f"📅 Removing models older than {older_than_days} days")

    models_dir = config.models_dir
    cutoff_date = datetime.now() - timedelta(days=older_than_days)

    if not models_dir.exists():
        console.print("❌ No models directory found", style="red")
        return

    # Find old model files
    old_files = []
    for model_file in models_dir.glob("*.joblib"):
        if datetime.fromtimestamp(model_file.stat().st_ctime) < cutoff_date:
            old_files.append(model_file)

    if not old_files:
        console.print("✅ No old models found to clean up", style="green")
        return

    console.print(f"📂 Found {len(old_files)} old model files:")

    for file_path in old_files[:10]:  # Show first 10
        created_date = datetime.fromtimestamp(file_path.stat().st_ctime)
        file_size = file_path.stat().st_size / (1024 * 1024)
        console.print(
            f"   • {file_path.name} - {created_date.strftime('%Y-%m-%d')} ({file_size:.1f}MB)"
        )

    if len(old_files) > 10:
        console.print(f"   ... and {len(old_files) - 10} more files")

    # Confirm deletion
    if not confirm:
        confirm = typer.confirm(f"\n🗑️  Delete {len(old_files)} old model files?")

    if confirm:
        deleted_count = 0
        total_size = 0

        for file_path in old_files:
            try:
                file_size = file_path.stat().st_size
                file_path.unlink()
                deleted_count += 1
                total_size += file_size
            except Exception as e:
                console.print(f"❌ Failed to delete {file_path.name}: {e}", style="red")

        total_size_mb = total_size / (1024 * 1024)
        console.print(
            f"\n✅ Deleted {deleted_count} files, freed {total_size_mb:.1f}MB",
            style="green",
        )
    else:
        console.print("❌ Cleanup cancelled", style="yellow")


@app.command()
def benchmark(
    symbol: str = typer.Option("TCS", help="Stock symbol for benchmarking"),
    iterations: int = typer.Option(5, help="Number of benchmark iterations"),
    models: List[str] = typer.Option(MODEL_TYPES, help="Models to benchmark"),
) -> None:
    """
    🏃‍♂️ Benchmark model training performance.
    """

    console.print("\n🏃‍♂️ [bold blue]Training Performance Benchmark[/bold blue]")
    console.print(f"📊 Stock: [cyan]{symbol.upper()}[/cyan]")
    console.print(f"🔄 Iterations: [yellow]{iterations}[/yellow]")
    console.print(f"🤖 Models: {', '.join(models)}")
    console.print()

    trainer = ModelTrainer()
    benchmark_results = {}

    for model_type in models:
        console.print(f"⏱️  Benchmarking [bold green]{model_type}[/bold green]...")

        times = []
        accuracies = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running {model_type}...", total=iterations)

            for i in range(iterations):
                progress.update(task, description=f"Iteration {i+1}/{iterations}")

                result = trainer.train_single_stock(
                    symbol=symbol.upper(),
                    model_type=model_type,
                    period="1y",  # Use smaller dataset for benchmarking
                    save_model=False,
                    optimize_hyperparams=False,
                    cross_validate=False,
                )

                if result["status"] == "success":
                    times.append(result["training_time"])
                    accuracies.append(result["metrics"]["accuracy"])

                progress.advance(task)

        if times:
            benchmark_results[model_type] = {
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "avg_accuracy": sum(accuracies) / len(accuracies),
                "successful_runs": len(times),
                "total_runs": iterations,
            }

    # Display benchmark results
    console.print(f"\n📊 [bold]Benchmark Results for {symbol.upper()}:[/bold]")

    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Avg Time (s)", justify="right")
    table.add_column("Min Time (s)", justify="right")
    table.add_column("Max Time (s)", justify="right")
    table.add_column("Avg Accuracy", justify="right")
    table.add_column("Success Rate", justify="right")

    for model_type, stats in benchmark_results.items():
        success_rate = (stats["successful_runs"] / stats["total_runs"]) * 100

        table.add_row(
            model_type,
            f"{stats['avg_time']:.2f}",
            f"{stats['min_time']:.2f}",
            f"{stats['max_time']:.2f}",
            f"{stats['avg_accuracy']:.1f}%",
            f"{success_rate:.0f}%",
        )

    console.print(table)

    # Performance recommendations
    if benchmark_results:
        fastest_model = min(benchmark_results.items(), key=lambda x: x[1]["avg_time"])
        most_accurate = max(benchmark_results.items(), key=lambda x: x[1]["avg_accuracy"])

        console.print("\n🏆 [bold]Performance Summary:[/bold]")
        console.print(
            f"   • Fastest: [green]{fastest_model[0]}[/green] ({fastest_model[1]['avg_time']:.2f}s)"
        )
        console.print(
            f"   • Most Accurate: [green]{most_accurate[0]}[/green] ({most_accurate[1]['avg_accuracy']:.1f}%)"
        )


if __name__ == "__main__":
    # Setup logging
    logger.add(
        config.logs_dir / "training_{time}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
    )

    app()
