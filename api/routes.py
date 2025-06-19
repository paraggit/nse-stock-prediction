"""
FastAPI routes for the NSE Stock Prediction API.

This module defines all the API endpoints for:
- Stock analysis and prediction
- Technical analysis
- Model management
- Portfolio analysis
- Health monitoring
- Market status

"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from api.models import (
    ComparisonRequest,
    ComparisonResponse,
    HealthResponse,
    MarketStatusResponse,
    ModelTrainingResponse,
    OutputFormat,
    PredictionResponse,
    StockAnalysis,
    StockInfo,
    TechnicalAnalysisResponse,
)
from modules.prediction_model import StockPredictionModel
from modules.technical_indicators import TechnicalIndicators

from .dependencies import (
    authenticate_request,
    cache_response,
    check_dependencies_health,
    check_rate_limit,
    get_market_status,
    get_popular_stocks,
    get_sector_stocks,
    get_stock_analyzer,
    log_request_middleware,
    monitor_performance,
    performance_monitor,
    return_stock_analyzer,
    validate_model_type,
    validate_period,
    validate_stock_symbol,
    validate_symbols_list,
)
from .models import (
    BatchAnalysisRequest,
    ModelMetrics,
    ModelTrainingRequest,
    PortfolioAnalysisRequest,
    PredictionResult,
    TechnicalSignals,
)

# Create main router
router = APIRouter()

# Sub-routers for different functionality
stocks_router = APIRouter(prefix="/stocks", tags=["stocks"])
technical_router = APIRouter(prefix="/technical", tags=["technical"])
predictions_router = APIRouter(prefix="/predictions", tags=["predictions"])
models_router = APIRouter(prefix="/models", tags=["models"])
market_router = APIRouter(prefix="/market", tags=["market"])
portfolio_router = APIRouter(prefix="/portfolio", tags=["portfolio"])
health_router = APIRouter(prefix="/health", tags=["health"])


# Stock Analysis Endpoints
@stocks_router.get(
    "/{symbol}/analyze",
    response_model=StockAnalysis,
    summary="Complete Stock Analysis",
    description="Get comprehensive stock analysis including prediction, technical indicators, and trading signals.",
)
@monitor_performance("stock_analyze")
@cache_response(ttl=300, key_prefix="stock_analysis")
async def analyze_stock(
    symbol: str = Path(..., description="Stock symbol (e.g., TCS, RELIANCE)"),
    period: str = Query("1y", description="Data period for analysis"),
    model_type: str = Query("random_forest", description="ML model type"),
    include_prediction: bool = Query(True, description="Include price prediction"),
    include_technical: bool = Query(True, description="Include technical analysis"),
    include_info: bool = Query(True, description="Include basic stock info"),
    cache_results: bool = Query(True, description="Cache analysis results"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: None = Depends(check_rate_limit),
    request_info: Dict[str, Any] = Depends(log_request_middleware),
):
    """
    Perform comprehensive stock analysis with ML prediction and technical indicators.

    This endpoint provides:
    - Current stock price and basic information
    - Machine learning price prediction with confidence intervals
    - Technical analysis with 20+ indicators
    - Trading signals and recommendations
    - Model performance metrics

    """
    try:
        # Validate inputs
        symbol = await validate_stock_symbol(symbol)
        period = validate_period(period)
        model_type = validate_model_type(model_type)

        # Get analyzer
        analyzer = await get_stock_analyzer()

        try:
            # Perform analysis
            result = analyzer.analyze_stock(
                symbol=symbol,
                period=period,
                include_prediction=include_prediction,
                include_technical=include_technical,
                include_info=include_info,
            )

            if result.get("status") != "success":
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Analysis failed: {result.get('error', 'Unknown error')}",
                )

            # Convert to response model
            response_data = StockAnalysis(
                symbol=symbol,
                analysis_date=datetime.now(),
                data_points=result.get("data_points", 0),
                stock_info=result.get("stock_info"),
                current_price=result.get("current_price", 0),
                prediction=result.get("prediction"),
                technical_indicators=result.get("technical_indicators"),
                technical_signals=result.get("technical_signals"),
                model_metrics=result.get("model_metrics"),
                feature_importance=result.get("feature_importance"),
            )

            # Log successful analysis
            background_tasks.add_task(
                logger.info,
                f"Stock analysis completed for {symbol} using {model_type} model",
            )

            return response_data

        finally:
            # Return analyzer to pool
            await return_stock_analyzer(analyzer)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stock analysis error for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal analysis error: {str(e)}",
        )


@stocks_router.get(
    "/{symbol}/info",
    response_model=StockInfo,
    summary="Stock Information",
    description="Get basic stock information including company details, market cap, and key metrics.",
)
@monitor_performance("stock_info")
@cache_response(ttl=600, key_prefix="stock_info")
async def get_stock_info(
    symbol: str = Path(..., description="Stock symbol"),
    _: None = Depends(check_rate_limit),
):
    """
    Get basic information about a stock.
    """
    try:
        symbol = await validate_stock_symbol(symbol)

        from modules.stock_data_fetcher import StockDataFetcher

        fetcher = StockDataFetcher()

        stock_info = fetcher.get_stock_info(symbol)

        if "error" in stock_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock information not found: {stock_info['error']}",
            )

        return StockInfo(**stock_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stock info error for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch stock information: {str(e)}",
        )


@stocks_router.post(
    "/compare",
    response_model=ComparisonResponse,
    summary="Compare Multiple Stocks",
    description="Compare multiple stocks across different metrics like predictions, technical signals, or performance.",
)
@monitor_performance("stock_compare")
async def compare_stocks(
    request: ComparisonRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: None = Depends(check_rate_limit),
):
    """
    Compare multiple stocks with specified analysis type.
    """
    try:
        # Validate symbols
        symbols = await validate_symbols_list(request.symbols)

        results = []
        analyzer = await get_stock_analyzer()

        try:
            # Analyze each stock
            for symbol in symbols:
                try:
                    result = analyzer.analyze_stock(
                        symbol=symbol,
                        period=request.period.value,
                        include_prediction=request.comparison_type in ["prediction", "performance"],
                        include_technical=request.comparison_type in ["technical", "performance"],
                        include_info=True,
                    )

                    if result.get("status") == "success":
                        results.append(result)
                    else:
                        logger.warning(f"Analysis failed for {symbol}: {result.get('error')}")

                except Exception as e:
                    logger.warning(f"Error analyzing {symbol}: {str(e)}")
                    continue

            if not results:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="No successful analyses for comparison",
                )

            # Generate summary based on comparison type
            summary = _generate_comparison_summary(results, request.comparison_type)

            return ComparisonResponse(
                symbols=[r["symbol"] for r in results],
                comparison_type=request.comparison_type,
                results=results,
                summary=summary,
            )

        finally:
            await return_stock_analyzer(analyzer)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stock comparison error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}",
        )


# Technical Analysis Endpoints
@technical_router.get(
    "/{symbol}",
    response_model=TechnicalAnalysisResponse,
    summary="Technical Analysis",
    description="Get technical analysis with indicators and trading signals.",
)
@monitor_performance("technical_analysis")
@cache_response(ttl=120, key_prefix="technical")
async def get_technical_analysis(
    symbol: str = Path(..., description="Stock symbol"),
    period: str = Query("6mo", description="Data period"),
    indicators: Optional[List[str]] = Query(None, description="Specific indicators"),
    _: None = Depends(check_rate_limit),
):
    """
    Get technical analysis for a stock.
    """
    try:
        symbol = await validate_stock_symbol(symbol)
        period = validate_period(period)

        from modules.stock_data_fetcher import StockDataFetcher

        # Fetch data
        fetcher = StockDataFetcher()
        data = fetcher.get_nse_stock_data(symbol, period)

        if data.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data available for {symbol}",
            )

        # Calculate technical indicators
        tech_analyzer = TechnicalIndicators()
        data_with_indicators = tech_analyzer.calculate_all_indicators(data)
        signals = tech_analyzer.get_latest_signals(data_with_indicators)

        # Get latest indicators
        latest = data_with_indicators.iloc[-1]

        technical_indicators = TechnicalIndicators(
            rsi=latest.get("RSI"),
            macd=latest.get("MACD"),
            macd_signal=latest.get("MACD_Signal"),
            macd_histogram=latest.get("MACD_Histogram"),
            bollinger_upper=latest.get("BB_Upper"),
            bollinger_lower=latest.get("BB_Lower"),
            bollinger_middle=latest.get("BB_Middle"),
            sma_20=latest.get("MA_20"),
            sma_50=latest.get("MA_50"),
            atr=latest.get("ATR"),
            stochastic_k=latest.get("Stoch_K"),
            williams_r=latest.get("Williams_R"),
            volume_sma=latest.get("Volume_MA"),
            price_change=latest.get("Price_Change"),
        )

        technical_signals = TechnicalSignals(
            rsi_signal=signals.get("rsi_signal", "Neutral"),
            macd_signal=signals.get("macd_signal", "Neutral"),
            ma_signal=signals.get("ma_signal", "Neutral"),
            bollinger_signal=signals.get("bb_signal", "Neutral"),
            volume_signal=signals.get("volume_signal", "Normal"),
            overall_signal=signals.get("overall_signal", "Neutral"),
        )

        return TechnicalAnalysisResponse(
            symbol=symbol,
            indicators=technical_indicators,
            signals=technical_signals,
            analysis_period=period,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Technical analysis error for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Technical analysis failed: {str(e)}",
        )


# Prediction Endpoints
@predictions_router.get(
    "/{symbol}",
    response_model=PredictionResponse,
    summary="Price Prediction",
    description="Get AI-powered price prediction with confidence intervals and model metrics.",
)
@monitor_performance("prediction")
@cache_response(ttl=180, key_prefix="prediction")
async def predict_stock_price(
    symbol: str = Path(..., description="Stock symbol"),
    period: str = Query("1y", description="Training data period"),
    model_type: str = Query("random_forest", description="ML model type"),
    target_days: int = Query(1, ge=1, le=30, description="Days ahead to predict"),
    include_confidence: bool = Query(True, description="Include confidence intervals"),
    retrain_model: bool = Query(False, description="Force model retraining"),
    _: None = Depends(check_rate_limit),
):
    """
    Get stock price prediction using machine learning models.
    """
    try:
        symbol = await validate_stock_symbol(symbol)
        period = validate_period(period)
        model_type = validate_model_type(model_type)

        analyzer = await get_stock_analyzer()

        try:
            # Configure analyzer model type
            analyzer.prediction_model = StockPredictionModel(model_type)

            # Get prediction
            result = analyzer.analyze_stock(
                symbol=symbol,
                period=period,
                include_prediction=True,
                include_technical=False,
                include_info=False,
            )

            if result.get("status") != "success":
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Prediction failed: {result.get('error', 'Unknown error')}",
                )

            prediction_data = result.get("prediction", {})
            model_metrics = result.get("model_metrics", {})

            return PredictionResponse(
                symbol=symbol,
                prediction=PredictionResult(**prediction_data),
                model_metrics=ModelMetrics(**model_metrics) if model_metrics else None,
            )

        finally:
            await return_stock_analyzer(analyzer)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        ) from e


# Model Management Endpoints
@models_router.post(
    "/train",
    response_model=ModelTrainingResponse,
    summary="Train ML Models",
    description="Train machine learning models for stock prediction.",
)
@monitor_performance("model_train")
async def train_models(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: None = Depends(check_rate_limit),
):
    """
    Train machine learning models for stock prediction.
    """
    try:
        # Use provided symbols or default to popular stocks
        symbols = request.symbols or get_popular_stocks()[:10]  # Limit to 10 for training

        # Validate symbols
        symbols = await validate_symbols_list(symbols, max_symbols=20)

        training_results = []
        total_start_time = datetime.now()

        analyzer = await get_stock_analyzer()

        try:
            for symbol in symbols:
                try:
                    start_time = datetime.now()

                    # Configure model
                    analyzer.prediction_model = StockPredictionModel(request.model_type.value)

                    # Train model
                    result = analyzer.analyze_stock(
                        symbol=symbol,
                        period=request.period.value,
                        include_prediction=True,
                        include_technical=False,
                        include_info=False,
                    )

                    training_time = (datetime.now() - start_time).total_seconds()

                    if result.get("status") == "success":
                        model_metrics = result.get("model_metrics", {})

                        training_result = {
                            "symbol": symbol,
                            "status": "success",
                            "model_type": request.model_type.value,
                            "accuracy": model_metrics.get("accuracy", 0),
                            "r2_score": model_metrics.get("test_r2", 0),
                            "training_time": training_time,
                            "data_points": result.get("data_points", 0),
                        }

                        # Save model if requested
                        if request.save_model:
                            model_path = (
                                f"models/saved_models/{symbol}_{request.model_type.value}.joblib"
                            )
                            analyzer.prediction_model.save_model(model_path)
                            training_result["model_saved"] = True

                        training_results.append(training_result)

                    else:
                        training_results.append(
                            {
                                "symbol": symbol,
                                "status": "failed",
                                "error": result.get("error", "Training failed"),
                                "training_time": training_time,
                            }
                        )

                except Exception as e:
                    logger.warning(f"Training failed for {symbol}: {str(e)}")
                    training_results.append(
                        {
                            "symbol": symbol,
                            "status": "failed",
                            "error": str(e),
                            "training_time": 0,
                        }
                    )

            # Calculate summary
            successful_trainings = [r for r in training_results if r["status"] == "success"]
            total_time = (datetime.now() - total_start_time).total_seconds()

            summary = {
                "total_symbols": len(symbols),
                "successful_trainings": len(successful_trainings),
                "failed_trainings": len(training_results) - len(successful_trainings),
                "average_accuracy": sum(r.get("accuracy", 0) for r in successful_trainings)
                / len(successful_trainings)
                if successful_trainings
                else 0,
                "total_training_time": total_time,
            }

            return ModelTrainingResponse(
                trained_models=training_results, summary=summary, total_time=total_time
            )

        finally:
            await return_stock_analyzer(analyzer)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model training error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(e)}",
        )


@models_router.get(
    "/list",
    summary="List Available Models",
    description="Get list of available and saved ML models.",
)
@monitor_performance("model_list")
async def list_models():
    """
    List available machine learning models.
    """
    try:
        import os
        from pathlib import Path

        available_models = ["random_forest", "gradient_boost", "linear_regression"]

        # Check for saved models
        saved_models = []
        models_dir = Path("models/saved_models")

        if models_dir.exists():
            for file_path in models_dir.glob("*.joblib"):
                file_name = file_path.stem
                parts = file_name.split("_")
                if len(parts) >= 2:
                    symbol = "_".join(parts[:-1])
                    model_type = parts[-1]

                    saved_models.append(
                        {
                            "symbol": symbol,
                            "model_type": model_type,
                            "file_path": str(file_path),
                            "file_size": os.path.getsize(file_path),
                            "created_date": datetime.fromtimestamp(
                                os.path.getctime(file_path)
                            ).isoformat(),
                        }
                    )

        return {
            "status": "success",
            "available_model_types": available_models,
            "saved_models": saved_models,
            "total_saved_models": len(saved_models),
        }

    except Exception as e:
        logger.error(f"Model list error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )


# Market Status Endpoints
@market_router.get(
    "/status",
    response_model=MarketStatusResponse,
    summary="Market Status",
    description="Get current NSE market status and trading hours.",
)
@monitor_performance("market_status")
@cache_response(ttl=60, key_prefix="market_status")
async def get_market_status_endpoint():
    """
    Get current NSE market status.
    """
    try:
        market_info = await get_market_status()

        return MarketStatusResponse(
            market_open=market_info["is_open"],
            market_status="OPEN" if market_info["is_open"] else "CLOSED",
            current_time=market_info["current_time"],
            next_open=market_info["next_open"],
            next_close=market_info["next_close"],
            timezone=market_info["timezone"],
        )

    except Exception as e:
        logger.error(f"Market status error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get market status: {str(e)}",
        )


@market_router.get(
    "/popular-stocks",
    summary="Popular Stocks",
    description="Get list of popular NSE stocks by sector.",
)
@cache_response(ttl=3600, key_prefix="popular_stocks")
async def get_popular_stocks_endpoint():
    """
    Get popular NSE stocks categorized by sector.
    """
    try:
        popular_stocks = get_popular_stocks()
        sector_stocks = get_sector_stocks()

        return {
            "status": "success",
            "popular_stocks": popular_stocks,
            "stocks_by_sector": sector_stocks,
            "total_stocks": len(popular_stocks),
        }

    except Exception as e:
        logger.error(f"Popular stocks error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get popular stocks: {str(e)}",
        )


# Portfolio Analysis Endpoints
@portfolio_router.post(
    "/analyze",
    summary="Portfolio Analysis",
    description="Analyze portfolio performance and risk metrics.",
)
@monitor_performance("portfolio_analysis")
async def analyze_portfolio(request: PortfolioAnalysisRequest, _: None = Depends(check_rate_limit)):
    """
    Analyze portfolio performance and risk.
    """
    try:
        # Get current prices for all stocks
        from modules.stock_data_fetcher import StockDataFetcher

        fetcher = StockDataFetcher()

        portfolio_analysis = []
        total_investment = 0
        total_current_value = 0

        for stock in request.stocks:
            try:
                # Get current price
                stock_info = fetcher.get_stock_info(stock.symbol)
                current_price = stock_info.get("current_price", 0)

                if current_price > 0:
                    investment = stock.quantity * stock.avg_price
                    current_value = stock.quantity * current_price
                    profit_loss = current_value - investment
                    profit_loss_percent = (profit_loss / investment) * 100 if investment > 0 else 0

                    stock_analysis = {
                        "symbol": stock.symbol,
                        "quantity": stock.quantity,
                        "avg_price": stock.avg_price,
                        "current_price": current_price,
                        "investment": investment,
                        "current_value": current_value,
                        "profit_loss": profit_loss,
                        "profit_loss_percent": profit_loss_percent,
                        "weight": 0,  # Will calculate after getting totals
                    }

                    portfolio_analysis.append(stock_analysis)
                    total_investment += investment
                    total_current_value += current_value

            except Exception as e:
                logger.warning(f"Error analyzing {stock.symbol}: {str(e)}")
                continue

        # Calculate weights
        for stock in portfolio_analysis:
            stock["weight"] = (
                (stock["current_value"] / total_current_value) * 100
                if total_current_value > 0
                else 0
            )

        # Portfolio summary
        total_profit_loss = total_current_value - total_investment
        total_profit_loss_percent = (
            (total_profit_loss / total_investment) * 100 if total_investment > 0 else 0
        )

        summary = {
            "total_investment": total_investment,
            "total_current_value": total_current_value,
            "total_profit_loss": total_profit_loss,
            "total_profit_loss_percent": total_profit_loss_percent,
            "number_of_stocks": len(portfolio_analysis),
            "best_performer": max(portfolio_analysis, key=lambda x: x["profit_loss_percent"])
            if portfolio_analysis
            else None,
            "worst_performer": min(portfolio_analysis, key=lambda x: x["profit_loss_percent"])
            if portfolio_analysis
            else None,
        }

        return {
            "status": "success",
            "portfolio_analysis": portfolio_analysis,
            "summary": summary,
            "analysis_type": request.analysis_type,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Portfolio analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Portfolio analysis failed: {str(e)}",
        )


# Batch Operations
@router.post(
    "/batch/analyze",
    summary="Batch Analysis",
    description="Analyze multiple stocks in batch with specified parameters.",
)
@monitor_performance("batch_analysis")
async def batch_analyze_stocks(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: None = Depends(check_rate_limit),
):
    """
    Perform batch analysis on multiple stocks.
    """
    try:
        symbols = await validate_symbols_list(request.symbols, max_symbols=50)

        results = []
        analyzer = await get_stock_analyzer()

        try:
            # Analyze stocks concurrently (but limit concurrency)
            semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent analyses

            async def analyze_single_stock(symbol: str):
                async with semaphore:
                    try:
                        result = analyzer.analyze_stock(
                            symbol=symbol,
                            period=request.period.value,
                            include_prediction=request.analysis_type in ["full", "prediction"],
                            include_technical=request.analysis_type in ["full", "technical"],
                            include_info=True,
                        )

                        if result.get("status") == "success":
                            return {
                                "symbol": symbol,
                                "status": "success",
                                "data": result,
                            }
                        else:
                            return {
                                "symbol": symbol,
                                "status": "failed",
                                "error": result.get("error", "Analysis failed"),
                            }

                    except Exception as e:
                        return {"symbol": symbol, "status": "failed", "error": str(e)}

            # Run analyses concurrently
            tasks = [analyze_single_stock(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and process results
            successful_results = []
            failed_results = []

            for result in results:
                if isinstance(result, Exception):
                    failed_results.append({"error": str(result)})
                elif result["status"] == "success":
                    successful_results.append(result)
                else:
                    failed_results.append(result)

            # Format response based on output format
            if request.output_format == OutputFormat.CSV:
                # Return CSV download link
                _format_results_as_csv(successful_results)
                # In a real implementation, save CSV and return download link
                return {
                    "status": "success",
                    "format": "csv",
                    "download_url": "/api/v1/downloads/batch_analysis.csv",
                    "successful_analyses": len(successful_results),
                    "failed_analyses": len(failed_results),
                }
            else:
                return {
                    "status": "success",
                    "format": "json",
                    "results": successful_results,
                    "failed_analyses": failed_results,
                    "summary": {
                        "total_requested": len(symbols),
                        "successful_analyses": len(successful_results),
                        "failed_analyses": len(failed_results),
                        "success_rate": (len(successful_results) / len(symbols)) * 100
                        if symbols
                        else 0,
                    },
                    "timestamp": datetime.now().isoformat(),
                }

        finally:
            await return_stock_analyzer(analyzer)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}",
        )


# Health and Monitoring Endpoints
@health_router.get(
    "/",
    response_model=HealthResponse,
    summary="System Health",
    description="Check overall system health and performance metrics.",
)
async def health_check():
    """
    Check system health and status.
    """
    try:
        # Check dependencies
        deps_health = await check_dependencies_health()

        # Get performance metrics
        perf_metrics = performance_monitor.get_metrics()

        return HealthResponse(
            api_version="1.0.0",
            system_status=deps_health,
            performance_metrics=perf_metrics,
            uptime=perf_metrics.get("uptime", 0),
        )

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}",
        )


@health_router.get(
    "/metrics",
    summary="Performance Metrics",
    description="Get detailed performance and usage metrics.",
)
async def get_metrics():
    """
    Get detailed system metrics.
    """
    try:
        metrics = performance_monitor.get_metrics()

        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}",
        )


# Helper functions
def _generate_comparison_summary(results: List[Dict], comparison_type: str) -> Dict[str, Any]:
    """
    Generate summary for stock comparison.
    """
    if comparison_type == "prediction":
        predictions = [r.get("prediction", {}) for r in results if r.get("prediction")]
        if not predictions:
            return {}

        avg_change = sum(p.get("percentage_change", 0) for p in predictions) / len(predictions)
        best_prediction = max(predictions, key=lambda x: x.get("percentage_change", 0))
        worst_prediction = min(predictions, key=lambda x: x.get("percentage_change", 0))

        return {
            "average_predicted_change": avg_change,
            "best_predicted_performer": best_prediction,
            "worst_predicted_performer": worst_prediction,
            "stocks_with_positive_outlook": len(
                [p for p in predictions if p.get("percentage_change", 0) > 0]
            ),
        }

    elif comparison_type == "technical":
        signals = [r.get("technical_signals", {}) for r in results if r.get("technical_signals")]
        if not signals:
            return {}

        bullish_count = len([s for s in signals if s.get("overall_signal") == "Bullish"])
        bearish_count = len([s for s in signals if s.get("overall_signal") == "Bearish"])
        neutral_count = len(signals) - bullish_count - bearish_count

        return {
            "bullish_signals": bullish_count,
            "bearish_signals": bearish_count,
            "neutral_signals": neutral_count,
            "market_sentiment": "Bullish"
            if bullish_count > bearish_count
            else "Bearish"
            if bearish_count > bullish_count
            else "Mixed",
        }

    return {}


def _format_results_as_csv(results: List[Dict]) -> str:
    """
    Format analysis results as CSV string.
    """
    if not results:
        return "No data available"

    # This is a simplified CSV formatter
    # In production, use pandas or csv module
    csv_lines = ["Symbol,Current Price,Predicted Price,Change %,Technical Signal"]

    for result in results:
        data = result.get("data", {})
        symbol = result["symbol"]
        current_price = data.get("current_price", 0)

        prediction = data.get("prediction", {})
        predicted_price = prediction.get("predicted_price", 0)
        change_percent = prediction.get("percentage_change", 0)

        technical_signals = data.get("technical_signals", {})
        overall_signal = technical_signals.get("overall_signal", "N/A")

        csv_lines.append(
            f"{symbol},{current_price},{predicted_price},{change_percent},{overall_signal}"
        )

    return "\n".join(csv_lines)


# Include all routers in main router
router.include_router(stocks_router)
router.include_router(technical_router)
router.include_router(predictions_router)
router.include_router(models_router)
router.include_router(market_router)
router.include_router(portfolio_router)
router.include_router(health_router)
