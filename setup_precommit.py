#!/usr/bin/env python3
"""
Setup script for pre-commit hooks in NSE Stock Prediction Pipeline.

This script:
1. Installs pre-commit hooks
2. Sets up baseline configurations
3. Runs initial checks
4. Provides helpful guidance

Run with: poetry run python setup_precommit.py

"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """
    Run a command and return success status and output.
    """
    try:
        print(f"🔄 {description}...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True, result.stdout
        else:
            print(f"❌ {description} - Failed")
            print(f"Error: {result.stderr}")
            return False, result.stderr

    except Exception as e:
        print(f"❌ {description} - Exception occurred: {str(e)}")
        return False, str(e)
