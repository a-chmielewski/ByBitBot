#!/usr/bin/env python3
"""
ByBit Trading Bot Setup Script
Handles complete environment setup, dependency installation, and initial configuration.
"""

import os
import sys
import subprocess
import platform
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional

# Minimum Python version required
MIN_PYTHON_VERSION = (3, 10)

class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message: str, color: str = Colors.GREEN) -> None:
    """Print colored message to terminal"""
    print(f"{color}{message}{Colors.END}")

def print_header(message: str) -> None:
    """Print header message"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}")
    print(f"{message}")
    print(f"{'='*60}{Colors.END}\n")

def print_step(step: int, total: int, message: str) -> None:
    """Print step message with progress indicator"""
    print(f"{Colors.BLUE}[{step}/{total}]{Colors.END} {message}")

def check_python_version() -> None:
    """Check if Python version meets minimum requirements"""
    print_header("CHECKING PYTHON VERSION")
    
    current_version = sys.version_info[:2]
    print(f"Current Python version: {current_version[0]}.{current_version[1]}")
    print(f"Required Python version: {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+")
    
    if current_version < MIN_PYTHON_VERSION:
        print_colored(f"ERROR: Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ is required!", Colors.RED)
        print_colored("Please upgrade your Python version and try again.", Colors.RED)
        sys.exit(1)
    
    print_colored("✓ Python version check passed", Colors.GREEN)

def check_system_requirements() -> None:
    """Check system-specific requirements"""
    print_header("CHECKING SYSTEM REQUIREMENTS")
    
    system = platform.system()
    print(f"Operating System: {system}")
    
    # Check for required system packages
    if system == "Windows":
        print_colored("Windows detected - TA-Lib installation may require Visual C++ Build Tools", Colors.YELLOW)
        print_colored("If TA-Lib installation fails, please install Microsoft C++ Build Tools", Colors.YELLOW)
        print_colored("Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/", Colors.YELLOW)
    elif system == "Linux":
        print_colored("Linux detected - TA-Lib installation may require build-essential", Colors.YELLOW)
        print_colored("Run: sudo apt-get install build-essential", Colors.YELLOW)
    elif system == "Darwin":  # macOS
        print_colored("macOS detected - TA-Lib installation may require Xcode Command Line Tools", Colors.YELLOW)
        print_colored("Run: xcode-select --install", Colors.YELLOW)
    
    # Check available disk space
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (1024**3)
        if free_gb < 1:
            print_colored(f"WARNING: Low disk space available ({free_gb}GB). At least 1GB recommended.", Colors.YELLOW)
        else:
            print_colored(f"✓ Available disk space: {free_gb}GB", Colors.GREEN)
    except Exception:
        print_colored("Could not check disk space", Colors.YELLOW)
    
    print_colored("✓ System requirements check completed", Colors.GREEN)

def create_virtual_environment() -> None:
    """Create and activate virtual environment"""
    print_header("SETTING UP VIRTUAL ENVIRONMENT")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print_colored("Virtual environment already exists", Colors.YELLOW)
        choice = input("Do you want to recreate it? (y/N): ").lower()
        if choice == 'y':
            print_colored("Removing existing virtual environment...", Colors.YELLOW)
            try:
                shutil.rmtree(venv_path)
                print_colored("✓ Existing virtual environment removed", Colors.GREEN)
            except Exception as e:
                print_colored(f"ERROR: Failed to remove existing virtual environment: {e}", Colors.RED)
                sys.exit(1)
        else:
            print_colored("Using existing virtual environment", Colors.GREEN)
            return
    
    print_colored("Creating virtual environment...", Colors.BLUE)
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True, capture_output=True)
        print_colored("✓ Virtual environment created successfully", Colors.GREEN)
    except subprocess.CalledProcessError as e:
        print_colored(f"ERROR: Failed to create virtual environment: {e}", Colors.RED)
        print_colored("Make sure you have the 'venv' module available", Colors.YELLOW)
        sys.exit(1)

def get_pip_command() -> str:
    """Get the appropriate pip command for the current system"""
    system = platform.system()
    if system == "Windows":
        return str(Path("venv/Scripts/pip.exe"))
    else:
        return str(Path("venv/bin/pip"))

def get_python_command() -> str:
    """Get the appropriate python command for the current system"""
    system = platform.system()
    if system == "Windows":
        return str(Path("venv/Scripts/python.exe"))
    else:
        return str(Path("venv/bin/python"))

def install_dependencies() -> None:
    """Install required dependencies"""
    print_header("INSTALLING DEPENDENCIES")
    
    pip_cmd = get_pip_command()
    
    # Check if pip exists
    if not Path(pip_cmd).exists():
        print_colored(f"ERROR: pip not found at {pip_cmd}", Colors.RED)
        sys.exit(1)
    
    # Upgrade pip first
    print_step(1, 4, "Upgrading pip...")
    try:
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True, capture_output=True)
        print_colored("✓ Pip upgraded successfully", Colors.GREEN)
    except subprocess.CalledProcessError as e:
        print_colored(f"WARNING: Failed to upgrade pip: {e}", Colors.YELLOW)
    
    # Install wheel for better package compilation
    print_step(2, 4, "Installing wheel...")
    try:
        subprocess.run([pip_cmd, "install", "wheel"], check=True, capture_output=True)
        print_colored("✓ Wheel installed successfully", Colors.GREEN)
    except subprocess.CalledProcessError as e:
        print_colored(f"WARNING: Failed to install wheel: {e}", Colors.YELLOW)
    
    # Install setuptools for better package handling
    print_step(3, 4, "Installing setuptools...")
    try:
        subprocess.run([pip_cmd, "install", "--upgrade", "setuptools"], check=True, capture_output=True)
        print_colored("✓ Setuptools installed successfully", Colors.GREEN)
    except subprocess.CalledProcessError as e:
        print_colored(f"WARNING: Failed to install setuptools: {e}", Colors.YELLOW)
    
    # Install dependencies from requirements.txt
    print_step(4, 4, "Installing dependencies from requirements.txt...")
    try:
        result = subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print_colored("✓ All dependencies installed successfully", Colors.GREEN)
        else:
            print_colored("WARNING: Some dependencies may have failed to install", Colors.YELLOW)
            print_colored("Error output:", Colors.YELLOW)
            print(result.stderr)
            
            # Special handling for TA-Lib
            if "TA-Lib" in result.stderr or "ta-lib" in result.stderr:
                print_colored("\nTA-Lib installation failed. Please install it manually:", Colors.YELLOW)
                system = platform.system()
                if system == "Windows":
                    print_colored("1. Download TA-Lib wheel from: https://github.com/cgohlke/talib-build/releases", Colors.YELLOW)
                    print_colored("2. Install with: pip install <downloaded-wheel-file>", Colors.YELLOW)
                elif system == "Linux":
                    print_colored("1. Install TA-Lib C library: sudo apt-get install libta-lib-dev", Colors.YELLOW)
                    print_colored("2. Then run: pip install ta-lib", Colors.YELLOW)
                elif system == "Darwin":  # macOS
                    print_colored("1. Install TA-Lib with Homebrew: brew install ta-lib", Colors.YELLOW)
                    print_colored("2. Then run: pip install ta-lib", Colors.YELLOW)
                
                # Continue with setup even if TA-Lib fails
                print_colored("Continuing with setup...", Colors.YELLOW)
                
    except FileNotFoundError:
        print_colored(f"ERROR: requirements.txt not found", Colors.RED)
        sys.exit(1)
    except Exception as e:
        print_colored(f"ERROR: Failed to install dependencies: {e}", Colors.RED)
        sys.exit(1)

def setup_directories() -> None:
    """Create necessary directories"""
    print_header("SETTING UP DIRECTORIES")
    
    directories = [
        "logs",
        "performance",
        "sessions/active",
        "sessions/archive", 
        "sessions/analysis",
        "test_sessions/active",
        "test_sessions/archive",
        "test_sessions/analysis",
        "config",
        "backtests",
        "data",
        "exports"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print_colored(f"✓ Created directory: {directory}", Colors.GREEN)
        else:
            print_colored(f"Directory already exists: {directory}", Colors.YELLOW)

def create_config_template() -> None:
    """Create configuration template"""
    print_header("CREATING CONFIGURATION TEMPLATE")
    
    config_path = Path("config/config.json")
    
    if config_path.exists():
        print_colored("Configuration file already exists", Colors.YELLOW)
        return
    
    config_template = {
        "bybit": {
            "api_key": "your_api_key_here",
            "api_secret": "your_api_secret_here",
            "testnet": True,
            "demo": True,
            "sandbox": True
        },
        "default": {
            "coin_pair": "BTC/USDT",
            "leverage": 10,
            "timeframe": "1m",
            "retry_attempts": 3,
            "retry_delay_seconds": 5,
            "max_position_size": 0.01,
            "risk_per_trade": 0.02,
            "max_daily_loss": 0.05,
            "max_drawdown": 0.10
        },
        "logging": {
            "level": "INFO",
            "max_file_size_mb": 10,
            "backup_count": 5,
            "console_output": True,
            "file_output": True
        },
        "performance": {
            "save_interval_minutes": 5,
            "backup_interval_hours": 24,
            "export_formats": ["csv", "json"],
            "track_metrics": True
        },
        "risk_management": {
            "enable_trailing_stops": True,
            "enable_position_sizing": True,
            "enable_drawdown_protection": True,
            "max_leverage": 50,
            "min_risk_reward_ratio": 1.5
        },
        "market_analysis": {
            "analysis_interval_minutes": 15,
            "symbols_to_analyze": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
            "timeframes": ["1m", "5m"],
            "enable_auto_strategy_selection": True
        }
    }
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config_template, f, indent=2)
        print_colored("✓ Configuration template created at config/config.json", Colors.GREEN)
        print_colored("⚠ Please update the API keys before running the bot!", Colors.YELLOW)
    except Exception as e:
        print_colored(f"ERROR: Failed to create configuration template: {e}", Colors.RED)

def create_env_template() -> None:
    """Create .env template file"""
    print_header("CREATING ENVIRONMENT TEMPLATE")
    
    env_path = Path(".env.template")
    
    if env_path.exists():
        print_colored("Environment template already exists", Colors.YELLOW)
        return
    
    env_template = """# ByBit Trading Bot Environment Configuration
# Copy this file to .env and fill in your actual values

# ByBit API Configuration
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
BYBIT_TESTNET=true
BYBIT_SANDBOX=true

# Trading Configuration
DEFAULT_COIN_PAIR=BTC/USDT
DEFAULT_LEVERAGE=10
DEFAULT_TIMEFRAME=1m

# Risk Management
MAX_POSITION_SIZE=0.01
RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.05
MAX_DRAWDOWN=0.10

# Logging
LOG_LEVEL=INFO
LOG_TO_CONSOLE=true
LOG_TO_FILE=true

# Performance Tracking
SAVE_INTERVAL_MINUTES=5
EXPORT_FORMATS=csv,json

# Market Analysis
ANALYSIS_INTERVAL_MINUTES=15
AUTO_STRATEGY_SELECTION=true
"""
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_template)
        print_colored("✓ Environment template created at .env.template", Colors.GREEN)
        print_colored("⚠ Copy to .env and update with your actual values!", Colors.YELLOW)
    except Exception as e:
        print_colored(f"ERROR: Failed to create environment template: {e}", Colors.RED)

def create_run_scripts() -> None:
    """Create platform-specific run scripts"""
    print_header("CREATING RUN SCRIPTS")
    
    # Windows batch script
    windows_script = """@echo off
echo ========================================
echo    ByBit Trading Bot
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\\Scripts\\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.py first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\\Scripts\\activate.bat

REM Check if bot.py exists
if not exist "bot.py" (
    echo ERROR: bot.py not found!
    echo Please make sure you're in the correct directory.
    pause
    exit /b 1
)

REM Run the bot
echo Starting ByBit Trading Bot...
python bot.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Bot stopped with an error.
    pause
)
"""
    
    # Linux/Mac shell script
    unix_script = """#!/bin/bash

echo "========================================"
echo "    ByBit Trading Bot"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run setup.py first."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if bot.py exists
if [ ! -f "bot.py" ]; then
    echo "ERROR: bot.py not found!"
    echo "Please make sure you're in the correct directory."
    exit 1
fi

# Run the bot
echo "Starting ByBit Trading Bot..."
python bot.py

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "Bot stopped with an error."
    read -p "Press Enter to continue..."
fi
"""
    
    # Development script for Windows
    dev_windows_script = """@echo off
echo ========================================
echo    ByBit Trading Bot - Development
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\\Scripts\\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.py first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\\Scripts\\activate.bat

REM Run tests
echo Running tests...
python -m pytest tests/ -v

REM Run the bot in development mode
echo Starting ByBit Trading Bot in development mode...
python bot.py --dev
"""
    
    # Development script for Unix
    dev_unix_script = """#!/bin/bash

echo "========================================"
echo "    ByBit Trading Bot - Development"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run setup.py first."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Run tests
echo "Running tests..."
python -m pytest tests/ -v

# Run the bot in development mode
echo "Starting ByBit Trading Bot in development mode..."
python bot.py --dev
"""
    
    try:
        # Create Windows scripts
        with open("run_bot.bat", 'w') as f:
            f.write(windows_script)
        print_colored("✓ Windows run script created: run_bot.bat", Colors.GREEN)
        
        with open("dev_bot.bat", 'w') as f:
            f.write(dev_windows_script)
        print_colored("✓ Windows development script created: dev_bot.bat", Colors.GREEN)
        
        # Create Unix scripts
        with open("run_bot.sh", 'w') as f:
            f.write(unix_script)
        
        with open("dev_bot.sh", 'w') as f:
            f.write(dev_unix_script)
        
        # Make Unix scripts executable
        if platform.system() != "Windows":
            os.chmod("run_bot.sh", 0o755)
            os.chmod("dev_bot.sh", 0o755)
        
        print_colored("✓ Unix run script created: run_bot.sh", Colors.GREEN)
        print_colored("✓ Unix development script created: dev_bot.sh", Colors.GREEN)
        
    except Exception as e:
        print_colored(f"ERROR: Failed to create run scripts: {e}", Colors.RED)

def create_gitignore() -> None:
    """Create .gitignore file"""
    print_header("CREATING .GITIGNORE")
    
    gitignore_path = Path(".gitignore")
    
    if gitignore_path.exists():
        print_colored(".gitignore already exists", Colors.YELLOW)
        return
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/

# Environment Variables
.env
.env.local
.env.production

# Configuration (contains API keys)
config/config.json
config/*.json

# Logs
logs/
*.log

# Performance Data
performance/
sessions/
test_sessions/

# Data Files
data/
exports/
backtests/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Temporary Files
*.tmp
*.temp
*.bak

# Test Coverage
.coverage
htmlcov/
.pytest_cache/

# Jupyter Notebooks
.ipynb_checkpoints/

# Trading Bot Specific
*.session
*.state
"""
    
    try:
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print_colored("✓ .gitignore created", Colors.GREEN)
    except Exception as e:
        print_colored(f"ERROR: Failed to create .gitignore: {e}", Colors.RED)

def verify_installation() -> None:
    """Verify that the installation was successful"""
    print_header("VERIFYING INSTALLATION")
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print_colored("ERROR: Virtual environment not found!", Colors.RED)
        return False
    
    # Check if bot.py exists
    if not Path("bot.py").exists():
        print_colored("ERROR: bot.py not found!", Colors.RED)
        return False
    
    # Check if config directory exists
    if not Path("config").exists():
        print_colored("ERROR: config directory not found!", Colors.RED)
        return False
    
    # Test Python import
    python_cmd = get_python_command()
    try:
        result = subprocess.run([python_cmd, "-c", "import sys; print('Python OK')"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_colored("✓ Python environment working", Colors.GREEN)
        else:
            print_colored("ERROR: Python environment not working", Colors.RED)
            return False
    except Exception as e:
        print_colored(f"ERROR: Could not test Python environment: {e}", Colors.RED)
        return False
    
    # Test basic imports
    test_imports = [
        "import pandas",
        "import numpy", 
        "import requests",
        "import logging"
    ]
    
    for import_stmt in test_imports:
        try:
            result = subprocess.run([python_cmd, "-c", import_stmt], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print_colored(f"✓ {import_stmt} - OK", Colors.GREEN)
            else:
                print_colored(f"WARNING: {import_stmt} - Failed", Colors.YELLOW)
        except Exception:
            print_colored(f"WARNING: Could not test {import_stmt}", Colors.YELLOW)
    
    print_colored("✓ Installation verification completed", Colors.GREEN)
    return True

def print_completion_message() -> None:
    """Print setup completion message"""
    print_header("SETUP COMPLETED SUCCESSFULLY!")
    
    print_colored("🎉 ByBit Trading Bot setup is complete!", Colors.GREEN)
    print_colored("\nNext steps:", Colors.BOLD)
    print_colored("1. Update config/config.json with your ByBit API credentials", Colors.YELLOW)
    print_colored("2. Copy .env.template to .env and configure your settings", Colors.YELLOW)
    print_colored("3. Run the bot using:", Colors.YELLOW)
    
    system = platform.system()
    if system == "Windows":
        print_colored("   - Production: run_bot.bat", Colors.BLUE)
        print_colored("   - Development: dev_bot.bat", Colors.BLUE)
        print_colored("   - Manual: venv\\Scripts\\activate && python bot.py", Colors.BLUE)
    else:
        print_colored("   - Production: ./run_bot.sh", Colors.BLUE)
        print_colored("   - Development: ./dev_bot.sh", Colors.BLUE)
        print_colored("   - Manual: source venv/bin/activate && python bot.py", Colors.BLUE)
    
    print_colored("\n⚠ IMPORTANT SECURITY NOTES:", Colors.RED)
    print_colored("- Never commit your API keys to version control", Colors.RED)
    print_colored("- Start with testnet/sandbox mode before live trading", Colors.RED)
    print_colored("- Test thoroughly with small amounts first", Colors.RED)
    print_colored("- Monitor the bot regularly during operation", Colors.RED)
    
    print_colored("\n📚 Documentation:", Colors.BLUE)
    print_colored("- README.md: Complete setup and usage guide", Colors.BLUE)
    print_colored("- Strategy files: Individual strategy documentation", Colors.BLUE)
    print_colored("- Tests: Run 'python -m pytest tests/' for testing", Colors.BLUE)
    
    print_colored(f"\n{'='*60}", Colors.BLUE)

def main() -> None:
    """Main setup function"""
    print_header("BYBIT TRADING BOT SETUP")
    print_colored("Setting up your cryptocurrency trading bot environment...", Colors.BLUE)
    
    try:
        # Run setup steps
        check_python_version()
        check_system_requirements()
        create_virtual_environment()
        install_dependencies()
        setup_directories()
        create_config_template()
        create_env_template()
        create_run_scripts()
        create_gitignore()
        
        # Verify installation
        if verify_installation():
            print_completion_message()
        else:
            print_colored("\n⚠ Setup completed with warnings. Please check the issues above.", Colors.YELLOW)
            print_completion_message()
        
    except KeyboardInterrupt:
        print_colored("\n\nSetup cancelled by user", Colors.YELLOW)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\nUnexpected error during setup: {e}", Colors.RED)
        print_colored("Please check the error message and try again.", Colors.YELLOW)
        sys.exit(1)

if __name__ == "__main__":
    main() 