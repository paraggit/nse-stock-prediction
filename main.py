#!/usr/bin/env python3
"""
Main application entry point for NSE Stock Prediction Pipeline Can be run as API server or CLI.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import configuration
try:
    from config import settings
except ImportError:
    print("Warning: Configuration not available, using defaults")

    class Settings:
        environment = "development"
        debug = False
        api_host = "0.0.0.0"
        api_port = 8000
        api_workers = 1
        api_reload = True

    settings = Settings()

# Check for dependencies
TYPER_AVAILABLE = False
FASTAPI_AVAILABLE = False

try:
    import typer
    from rich.console import Console

    TYPER_AVAILABLE = True
    console = Console()
except ImportError:
    print("Warning: typer/rich not available, using basic interface")

try:
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    print("Warning: FastAPI/uvicorn not available, API mode disabled")


def run_api_server():
    """
    Run the FastAPI server.
    """
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available. Install with: poetry add fastapi uvicorn")
        return False

    try:
        # Import API application
        from api.main import app

        print(f"🚀 Starting {getattr(settings, 'app_name', 'NSE Stock Prediction API')}")
        print(f"🌐 Server: http://{settings.api_host}:{settings.api_port}")
        print(f"📚 API Docs: http://{settings.api_host}:{settings.api_port}/docs")
        print()

        # Run server
        uvicorn.run(
            app,
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.api_reload and settings.environment == "development",
            workers=settings.api_workers if not settings.api_reload else 1,
            log_level=getattr(settings, "log_level", "info").lower(),
        )
        return True

    except ImportError as e:
        print(f"❌ Could not import API application: {e}")
        print("💡 Make sure FastAPI dependencies are installed: poetry add fastapi uvicorn")
        return False
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False


def run_cli_interface(args: List[str] = None):
    """
    Run the CLI interface.
    """
    if not TYPER_AVAILABLE:
        return run_simple_cli(args)

    try:
        from cli import app as cli_app

        # If no args provided, show help
        if not args:
            args = ["--help"]

        # Update sys.argv for the CLI app
        original_argv = sys.argv.copy()
        sys.argv = ["stock-cli"] + args

        try:
            cli_app()
        finally:
            # Restore original argv
            sys.argv = original_argv

    except ImportError as e:
        print(f"❌ Could not import CLI application: {e}")
        return run_simple_cli(args)
    except Exception as e:
        print(f"❌ CLI error: {e}")
        return run_simple_cli(args)


def run_simple_cli(args: List[str] = None):
    """
    Simple CLI fallback without typer.
    """

    print("NSE Stock Prediction Pipeline")
    print("=" * 50)

    if not args or args == ["--help"] or args == ["-h"]:
        print("Usage: python main.py [command] [options]")
        print()
        print("Commands:")
        print("  api                  - Start FastAPI server")
        print("  cli [args]          - Run CLI interface")
        print("  analyze <symbol>    - Quick stock analysis")
        print("  info <symbol>       - Get stock information")
        print("  health             - System health check")
        print("  version            - Show version")
        print("  help               - Show this help")
        print()
        print("Examples:")
        print("  python main.py api")
        print("  python main.py analyze TCS")
        print("  python main.py info RELIANCE")
        return

    command = args[0].lower() if args else "help"

    if command == "api":
        run_api_server()
    elif command == "analyze" and len(args) > 1:
        quick_analyze(args[1])
    elif command == "info" and len(args) > 1:
        quick_info(args[1])
    elif command == "health":
        system_health_check()
    elif command == "version":
        show_version()
    elif command in ["help", "--help", "-h"]:
        run_simple_cli(["--help"])
    else:
        print(f"Unknown command: {command}")
        print("Use 'python main.py help' for available commands")


def quick_analyze(symbol: str):
    """
    Quick stock analysis without full CLI.
    """
    try:
        from modules.stock_analyzer import StockAnalyzer

        print(f"\n🔍 Analyzing {symbol.upper()}...")

        analyzer = StockAnalyzer()
        result = analyzer.analyze_stock(symbol.upper(), period="3mo")

        if result["status"] == "success":
            print(f"\n✅ Analysis completed for {symbol.upper()}")
            print(f"Current Price: ₹{result.get('current_price', 'N/A'):.2f}")

            if "prediction" in result:
                pred = result["prediction"]
                print(f"Predicted Price: ₹{pred['predicted_price']:.2f}")
                print(f"Expected Change: {pred['percentage_change']:.2f}%")

            if "technical_signals" in result:
                signals = result["technical_signals"]
                print(f"Technical Signal: {signals.get('overall_signal', 'N/A')}")

            if "model_metrics" in result:
                metrics = result["model_metrics"]
                print(f"Model Accuracy: {metrics.get('accuracy', 0):.1f}%")
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


def quick_info(symbol: str):
    """
    Quick stock info without full CLI.
    """
    try:
        from modules.stock_analyzer import StockAnalyzer

        print(f"\n📊 Getting info for {symbol.upper()}...")

        analyzer = StockAnalyzer()
        result = analyzer.get_stock_info_only(symbol.upper())

        if result["status"] == "success":
            info = result["stock_info"]
            print(f"\n✅ Stock Information for {symbol.upper()}")
            print(f"Company: {info.get('company_name', 'N/A')}")
            print(f"Sector: {info.get('sector', 'N/A')}")
            print(f"Current Price: ₹{info.get('current_price', 0):.2f}")
            print(f"Market Cap: ₹{info.get('market_cap', 0):,}")
            print(f"P/E Ratio: {info.get('pe_ratio', 0):.2f}")
        else:
            print(f"❌ Failed to get info: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


def system_health_check():
    """
    Basic system health check.
    """
    print("\n🏥 System Health Check")
    print("=" * 30)

    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python Version: {python_version}")

    # Check dependencies
    dependencies = [
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing"),
        ("yfinance", "Stock data fetching"),
        ("sklearn", "Machine learning"),
    ]

    print("\n📦 Core Dependencies:")
    for dep_name, description in dependencies:
        try:
            __import__(dep_name)
            print(f"  ✅ {dep_name}: {description}")
        except ImportError:
            print(f"  ❌ {dep_name}: {description} - NOT INSTALLED")

    # Check optional dependencies
    optional_deps = [
        ("typer", "CLI interface"),
        ("rich", "Rich console output"),
        ("fastapi", "API framework"),
        ("uvicorn", "ASGI server"),
        ("ta", "Technical analysis"),
    ]

    print("\n📦 Optional Dependencies:")
    for dep_name, description in optional_deps:
        try:
            __import__(dep_name)
            print(f"  ✅ {dep_name}: {description}")
        except ImportError:
            print(f"  ⚠️  {dep_name}: {description} - NOT INSTALLED")

    # Check configuration
    print("\n⚙️  Configuration:")
    print(f"  📂 Environment: {getattr(settings, 'environment', 'unknown')}")
    print(f"  🔧 Debug mode: {getattr(settings, 'debug', 'unknown')}")

    # Check directories
    print("\n📁 Directories:")
    directories = [Path("models/saved_models"), Path("data/cache"), Path("logs")]

    for directory in directories:
        if directory.exists():
            print(f"  ✅ {directory}")
        else:
            print(f"  ⚠️  {directory} - MISSING")

    print("\n🎯 Overall Status: System operational with available dependencies")


def show_version():
    """
    Show version information.
    """
    print("\nNSE Stock Prediction Pipeline")
    print("Version: 1.0.0")
    print(f"Environment: {getattr(settings, 'environment', 'unknown')}")
    print(f"Python: {sys.version.split()[0]}")


# Simple argument parsing
def parse_args():
    """
    Parse command line arguments.
    """
    if len(sys.argv) <= 1:
        return []

    return sys.argv[1:]


def main():
    """
    Main entry point with intelligent mode detection.
    """

    # Parse arguments
    args = parse_args()

    # If no arguments, show help
    if not args:
        run_simple_cli(["--help"])
        return

    command = args[0].lower()

    # Handle different modes
    if command in ["api", "server"]:
        success = run_api_server()
        if not success:
            sys.exit(1)
    elif command == "cli":
        run_cli_interface(args[1:] if len(args) > 1 else [])
    elif command in ["analyze", "info", "health", "version", "help", "--help", "-h"]:
        run_simple_cli(args)
    else:
        # Try to run as CLI command
        run_cli_interface(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        if getattr(settings, "debug", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)
