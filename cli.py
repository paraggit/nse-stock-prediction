#!/usr/bin/env python3
"""
Main CLI entry point for NSE Stock Prediction Pipeline
Run with: poetry run stock-cli
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    print("Warning: typer and rich not available, using basic CLI")
    RICH_AVAILABLE = False

    # Simple fallback CLI
    def simple_cli():
        if len(sys.argv) < 2:
            print("NSE Stock Prediction Pipeline")
            print("Usage: python cli.py [command] [options]")
            print("\nCommands:")
            print("  analyze <symbol>  - Analyze a stock")
            print("  info <symbol>     - Get stock information")
            print("  help             - Show this help")
            return

        command = sys.argv[1].lower()

        if command == "help" or command == "--help":
            simple_cli()
        elif command == "analyze" and len(sys.argv) > 2:
            symbol = sys.argv[2].upper()
            simple_analyze(symbol)
        elif command == "info" and len(sys.argv) > 2:
            symbol = sys.argv[2].upper()
            simple_info(symbol)
        else:
            print(f"Unknown command: {command}")
            simple_cli()

    def simple_analyze(symbol: str):
        try:
            from modules.stock_analyzer import StockAnalyzer

            print(f"\n🔍 Analyzing {symbol}...")

            analyzer = StockAnalyzer()
            result = analyzer.analyze_stock(symbol, period="3mo")

            if result["status"] == "success":
                print(f"\n✅ Analysis completed for {symbol}")
                print(f"Current Price: ₹{result.get('current_price', 'N/A'):.2f}")

                if "prediction" in result:
                    pred = result["prediction"]
                    print(f"Predicted Price: ₹{pred['predicted_price']:.2f}")
                    print(f"Expected Change: {pred['percentage_change']:.2f}%")

                if "technical_signals" in result:
                    signals = result["technical_signals"]
                    print(f"Technical Signal: {signals.get('overall_signal', 'N/A')}")
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

    def simple_info(symbol: str):
        try:
            from modules.stock_analyzer import StockAnalyzer

            print(f"\n📊 Getting info for {symbol}...")

            analyzer = StockAnalyzer()
            result = analyzer.get_stock_info_only(symbol)

            if result["status"] == "success":
                info = result["stock_info"]
                print(f"\n✅ Stock Information for {symbol}")
                print(f"Company: {info.get('company_name', 'N/A')}")
                print(f"Sector: {info.get('sector', 'N/A')}")
                print(f"Current Price: ₹{info.get('current_price', 0):.2f}")
                print(f"Market Cap: ₹{info.get('market_cap', 0):,}")
            else:
                print(f"❌ Failed to get info: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")


if not RICH_AVAILABLE:
    if __name__ == "__main__":
        simple_cli()
    sys.exit()

# Rich CLI implementation
console = Console()

# Create main Typer app with updated syntax
app = typer.Typer(
    name="stock-cli",
    help="🚀 NSE Stock Prediction Pipeline CLI",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def analyze(
    symbol: str = typer.Argument(..., help="Stock symbol (e.g., TCS, RELIANCE)"),
    period: str = typer.Option("1y", "--period", "-p", help="Data period"),
    model_type: str = typer.Option("random_forest", "--model", "-m", help="Model type"),
    save_result: bool = typer.Option(False, "--save", help="Save analysis result"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
) -> None:
    """
    📊 Analyze a single stock with prediction and technical analysis.
    """

    console.print(f"\n🔍 [bold blue]Analyzing {symbol.upper()}[/bold blue]...\n")

    try:
        from modules.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer(model_type=model_type)

        with console.status("[bold green]Running analysis..."):
            result = analyzer.analyze_stock(
                symbol=symbol.upper(),
                period=period,
                include_prediction=True,
                include_technical=True,
                include_info=True,
            )

        if result.get("status") == "success":
            display_analysis_result(result)

            if save_result or output:
                save_path = (
                    output
                    or f"{symbol.upper()}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(save_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                console.print(f"\n💾 Results saved to: [green]{save_path}[/green]")

        else:
            console.print(f"❌ Analysis failed: [red]{result.get('error', 'Unknown error')}[/red]")

    except ImportError as e:
        console.print(f"❌ Import error: [red]{str(e)}[/red]")
        console.print("💡 Make sure all dependencies are installed: [yellow]poetry install[/yellow]")
    except Exception as e:
        console.print(f"❌ Error during analysis: [red]{str(e)}[/red]")


@app.command()
def predict(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    period: str = typer.Option("1y", "--period", "-p", help="Data period for training"),
    model_type: str = typer.Option("random_forest", "--model", "-m", help="Model type"),
) -> None:
    """
    🔮 Get price prediction for a stock.
    """

    console.print(f"\n🔮 [bold blue]Predicting price for {symbol.upper()}[/bold blue]...\n")

    try:
        from modules.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer(model_type=model_type)

        with console.status("[bold green]Training model and making prediction..."):
            result = analyzer.get_prediction_only(symbol.upper(), period)

        if result.get("status") == "success":
            display_prediction_result(result)
        else:
            console.print(f"❌ Prediction failed: [red]{result.get('error', 'Unknown error')}[/red]")

    except Exception as e:
        console.print(f"❌ Error during prediction: [red]{str(e)}[/red]")


@app.command()
def technical(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    period: str = typer.Option("6mo", "--period", "-p", help="Data period"),
) -> None:
    """
    📈 Get technical analysis for a stock.
    """

    console.print(f"\n📈 [bold blue]Technical analysis for {symbol.upper()}[/bold blue]...\n")

    try:
        from modules.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()

        with console.status("[bold green]Calculating technical indicators..."):
            result = analyzer.get_technical_analysis_only(symbol.upper(), period)

        if result.get("status") == "success":
            display_technical_result(result)
        else:
            console.print(
                f"❌ Technical analysis failed: [red]{result.get('error', 'Unknown error')}[/red]"
            )

    except Exception as e:
        console.print(f"❌ Error during technical analysis: [red]{str(e)}[/red]")


@app.command()
def info(symbol: str = typer.Argument(..., help="Stock symbol")) -> None:
    """
    ℹ️ Get basic information about a stock.
    """

    console.print(f"\nℹ️ [bold blue]Stock information for {symbol.upper()}[/bold blue]...\n")

    try:
        from modules.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        result = analyzer.get_stock_info_only(symbol.upper())

        if result.get("status") == "success":
            display_stock_info(result)
        else:
            console.print(
                f"❌ Failed to get stock info: [red]{result.get('error', 'Unknown error')}[/red]"
            )

    except Exception as e:
        console.print(f"❌ Error getting stock info: [red]{str(e)}[/red]")


@app.command()
def compare(
    symbols: List[str] = typer.Argument(..., help="Stock symbols to compare"),
    period: str = typer.Option("6mo", "--period", "-p", help="Data period"),
    analysis_type: str = typer.Option(
        "prediction", "--type", "-t", help="Analysis type: prediction, technical, info"
    ),
) -> None:
    """
    ⚖️ Compare multiple stocks.
    """

    symbols_upper = [s.upper() for s in symbols]
    console.print(f"\n⚖️ [bold blue]Comparing stocks: {', '.join(symbols_upper)}[/bold blue]\n")

    try:
        from modules.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()

        with console.status("[bold green]Analyzing stocks..."):
            result = analyzer.compare_stocks(symbols_upper, period, analysis_type)

        if result.get("status") == "success":
            display_comparison_result(result)
        else:
            console.print(f"❌ Comparison failed: [red]{result.get('error', 'Unknown error')}[/red]")

    except Exception as e:
        console.print(f"❌ Error during comparison: [red]{str(e)}[/red]")


@app.command()
def market_status() -> None:
    """
    🏛️ Check NSE market status and trading hours.
    """

    console.print("\n🏛️ [bold blue]NSE Market Status[/bold blue]\n")

    try:
        from datetime import datetime

        import pytz

        # IST timezone
        ist = pytz.timezone("Asia/Kolkata")
        current_time = datetime.now(ist)

        # Market hours (9:15 AM to 3:30 PM IST)
        market_open = current_time.replace(hour=9, minute=15, second=0)
        market_close = current_time.replace(hour=15, minute=30, second=0)

        # Check if it's a weekday
        is_weekday = current_time.weekday() < 5

        if is_weekday and market_open <= current_time <= market_close:
            status = "🟢 OPEN"
            status_color = "green"
        elif is_weekday and current_time < market_open:
            status = "🟡 PRE-MARKET"
            status_color = "yellow"
        elif is_weekday and current_time > market_close:
            status = "🔴 CLOSED"
            status_color = "red"
        else:
            status = "🔴 CLOSED (Weekend)"
            status_color = "red"

        table = Table()
        table.add_column("Item", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Market Status", f"[{status_color}]{status}[/{status_color}]")
        table.add_row("Current Time (IST)", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        table.add_row("Market Open", "09:15 IST")
        table.add_row("Market Close", "15:30 IST")
        table.add_row("Day", current_time.strftime("%A"))

        console.print(table)

    except ImportError:
        console.print("❌ pytz library required for timezone support")
    except Exception as e:
        console.print(f"❌ Error getting market status: [red]{str(e)}[/red]")


@app.command()
def version() -> None:
    """
    📋 Show version information.
    """

    console.print("\n📋 [bold blue]NSE Stock Prediction Pipeline[/bold blue]")
    console.print("Version: [green]1.0.0[/green]")
    console.print("Built with Poetry 📦")


# Helper functions for displaying results
def display_analysis_result(result: dict) -> None:
    """
    Display complete analysis result.
    """

    symbol = result["symbol"]
    current_price = result.get("current_price", 0)

    # Basic info
    console.print(f"📊 [bold]Stock: {symbol}[/bold]")
    console.print(f"💰 Current Price: [bold green]₹{current_price:.2f}[/bold green]")

    # Stock info
    if "stock_info" in result:
        info = result["stock_info"]
        console.print(f"🏢 Company: {info.get('company_name', 'N/A')}")
        console.print(f"🏭 Sector: {info.get('sector', 'N/A')}")

    console.print()

    # Prediction
    if "prediction" in result:
        pred = result["prediction"]
        change_color = "green" if pred["percentage_change"] >= 0 else "red"

        pred_table = Table(title="🔮 Price Prediction")
        pred_table.add_column("Metric", style="cyan")
        pred_table.add_column("Value", style="white")

        pred_table.add_row("Predicted Price", f"₹{pred['predicted_price']:.2f}")
        pred_table.add_row("Expected Change", f"₹{pred['price_change']:.2f}")
        pred_table.add_row(
            "Percentage Change",
            f"[{change_color}]{pred['percentage_change']:.2f}%[/{change_color}]",
        )

        console.print(pred_table)
        console.print()

    # Technical signals
    if "technical_signals" in result:
        signals = result["technical_signals"]

        tech_table = Table(title="📈 Technical Signals")
        tech_table.add_column("Indicator", style="cyan")
        tech_table.add_column("Signal", style="white")

        for indicator, signal in signals.items():
            if indicator != "overall_signal":
                tech_table.add_row(indicator.replace("_", " ").title(), signal)

        tech_table.add_row(
            "[bold]Overall Signal[/bold]",
            f"[bold]{signals.get('overall_signal', 'N/A')}[/bold]",
        )

        console.print(tech_table)

    # Model metrics
    if "model_metrics" in result:
        metrics = result["model_metrics"]
        console.print(
            f"\n🎯 Model Accuracy: [bold cyan]{metrics.get('accuracy', 0):.1f}%[/bold cyan]"
        )


def display_prediction_result(result: dict) -> None:
    """
    Display prediction result.
    """

    if "prediction" not in result:
        return

    prediction = result["prediction"]
    symbol = result["symbol"]

    table = Table(title=f"🔮 Price Prediction - {symbol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Current Price", f"₹{prediction['current_price']:.2f}")
    table.add_row("Predicted Price", f"₹{prediction['predicted_price']:.2f}")

    change_color = "green" if prediction["price_change"] >= 0 else "red"
    table.add_row(
        "Expected Change",
        f"[{change_color}]₹{prediction['price_change']:.2f}[/{change_color}]",
    )
    table.add_row(
        "Percentage Change",
        f"[{change_color}]{prediction['percentage_change']:.2f}%[/{change_color}]",
    )

    if "model_metrics" in result:
        table.add_row("Model Accuracy", f"{result['model_metrics'].get('accuracy', 0):.1f}%")

    console.print(table)


def display_technical_result(result: dict) -> None:
    """
    Display technical analysis result.
    """

    if "technical_signals" not in result:
        console.print("❌ No technical data available")
        return

    signals = result["technical_signals"]
    symbol = result["symbol"]

    table = Table(title=f"📈 Technical Analysis - {symbol}")
    table.add_column("Indicator", style="cyan")
    table.add_column("Signal", style="white")

    for indicator, signal in signals.items():
        if indicator != "overall_signal":
            table.add_row(indicator.replace("_", " ").title(), signal)

    # Overall signal
    overall_signal = signals.get("overall_signal", "N/A")
    table.add_row("[bold]Overall Signal[/bold]", f"[bold]{overall_signal}[/bold]")

    console.print(table)


def display_stock_info(result: dict) -> None:
    """
    Display stock information.
    """

    if "stock_info" not in result:
        console.print("❌ No stock information available")
        return

    stock_info = result["stock_info"]
    symbol = result["symbol"]

    table = Table(title=f"ℹ️ Stock Information - {symbol}")
    table.add_column("Attribute", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Company Name", stock_info.get("company_name", "N/A"))
    table.add_row("Sector", stock_info.get("sector", "N/A"))
    table.add_row("Current Price", f"₹{stock_info.get('current_price', 0):.2f}")
    table.add_row("Market Cap", f"₹{stock_info.get('market_cap', 0):,}")
    table.add_row("52 Week High", f"₹{stock_info.get('fifty_two_week_high', 0):.2f}")
    table.add_row("52 Week Low", f"₹{stock_info.get('fifty_two_week_low', 0):.2f}")
    table.add_row("P/E Ratio", f"{stock_info.get('pe_ratio', 0):.2f}")

    console.print(table)


def display_comparison_result(result: dict) -> None:
    """
    Display comparison of multiple stocks.
    """

    if "results" not in result:
        console.print("❌ No comparison data available")
        return

    results = result["results"]
    comparison_type = result.get("comparison_type", "prediction")

    if comparison_type == "prediction":
        table = Table(title="⚖️ Stock Prediction Comparison")
        table.add_column("Symbol", style="cyan")
        table.add_column("Current Price", justify="right")
        table.add_column("Predicted Price", justify="right")
        table.add_column("Change %", justify="right")

        for stock_result in results:
            if "prediction" in stock_result:
                pred = stock_result["prediction"]
                change_color = "green" if pred["percentage_change"] >= 0 else "red"

                table.add_row(
                    stock_result["symbol"],
                    f"₹{pred['current_price']:.2f}",
                    f"₹{pred['predicted_price']:.2f}",
                    f"[{change_color}]{pred['percentage_change']:.2f}%[/{change_color}]",
                )

    elif comparison_type == "technical":
        table = Table(title="⚖️ Technical Signal Comparison")
        table.add_column("Symbol", style="cyan")
        table.add_column("RSI Signal", style="white")
        table.add_column("MACD Signal", style="white")
        table.add_column("Overall Signal", style="white")

        for stock_result in results:
            if "technical_signals" in stock_result:
                signals = stock_result["technical_signals"]
                table.add_row(
                    stock_result["symbol"],
                    signals.get("rsi_signal", "N/A"),
                    signals.get("macd_signal", "N/A"),
                    f"[bold]{signals.get('overall_signal', 'N/A')}[/bold]",
                )

    console.print(table)

    # Summary
    if "summary" in result:
        summary = result["summary"]
        console.print(f"\n📊 Summary: {summary}")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n\n👋 Goodbye!", style="blue")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n❌ Unexpected error: [red]{str(e)}[/red]")
        sys.exit(1)
