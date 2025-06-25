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
from pathlib import Path

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

def print_colored(message, color=Colors.GREEN):
    """Print colored message to terminal"""
    print(f"{color}{message}{Colors.END}")

def print_header(message):
    """Print header message"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}")
    print(f"{message}")
    print(f"{'='*60}{Colors.END}\n")

def check_python_version():
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

def check_system_requirements():
    """Check system-specific requirements"""
    print_header("CHECKING SYSTEM REQUIREMENTS")
    
    system = platform.system()
    print(f"Operating System: {system}")
    
    # Check for required system packages
    if system == "Windows":
        print_colored("Windows detected - TA-Lib installation may require Visual C++ Build Tools", Colors.YELLOW)
        print_colored("If TA-Lib installation fails, please install Microsoft C++ Build Tools", Colors.YELLOW)
    elif system == "Linux":
        print_colored("Linux detected - TA-Lib installation may require build-essential", Colors.YELLOW)
        print_colored("Run: sudo apt-get install build-essential", Colors.YELLOW)
    
    print_colored("✓ System requirements check completed", Colors.GREEN)

def create_virtual_environment():
    """Create and activate virtual environment"""
    print_header("SETTING UP VIRTUAL ENVIRONMENT")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print_colored("Virtual environment already exists", Colors.YELLOW)
        choice = input("Do you want to recreate it? (y/N): ").lower()
        if choice == 'y':
            print_colored("Removing existing virtual environment...", Colors.YELLOW)
            import shutil
            shutil.rmtree(venv_path)
        else:
            print_colored("Using existing virtual environment", Colors.GREEN)
            return
    
    print_colored("Creating virtual environment...", Colors.BLUE)
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print_colored("✓ Virtual environment created successfully", Colors.GREEN)
    except subprocess.CalledProcessError as e:
        print_colored(f"ERROR: Failed to create virtual environment: {e}", Colors.RED)
        sys.exit(1)

def get_pip_command():
    """Get the appropriate pip command for the current system"""
    system = platform.system()
    if system == "Windows":
        return str(Path("venv/Scripts/pip.exe"))
    else:
        return str(Path("venv/bin/pip"))

def install_dependencies():
    """Install required dependencies"""
    print_header("INSTALLING DEPENDENCIES")
    
    pip_cmd = get_pip_command()
    
    # Upgrade pip first
    print_colored("Upgrading pip...", Colors.BLUE)
    try:
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        print_colored("✓ Pip upgraded successfully", Colors.GREEN)
    except subprocess.CalledProcessError as e:
        print_colored(f"WARNING: Failed to upgrade pip: {e}", Colors.YELLOW)
    
    # Install wheel for better package compilation
    print_colored("Installing wheel...", Colors.BLUE)
    try:
        subprocess.run([pip_cmd, "install", "wheel"], check=True)
        print_colored("✓ Wheel installed successfully", Colors.GREEN)
    except subprocess.CalledProcessError as e:
        print_colored(f"WARNING: Failed to install wheel: {e}", Colors.YELLOW)
    
    # Install dependencies from requirements.txt
    print_colored("Installing dependencies from requirements.txt...", Colors.BLUE)
    try:
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        print_colored("✓ All dependencies installed successfully", Colors.GREEN)
    except subprocess.CalledProcessError as e:
        print_colored(f"ERROR: Failed to install dependencies: {e}", Colors.RED)
        print_colored("You may need to install some dependencies manually", Colors.YELLOW)
        
        # Special handling for TA-Lib
        if "TA-Lib" in str(e) or "ta-lib" in str(e):
            print_colored("\nTA-Lib installation failed. Please install it manually:", Colors.YELLOW)
            if platform.system() == "Windows":
                print_colored("1. Download TA-Lib wheel from: https://github.com/cgohlke/talib-build/releases", Colors.YELLOW)
                print_colored("2. Install with: pip install <downloaded-wheel-file>", Colors.YELLOW)
            else:
                print_colored("1. Install TA-Lib C library: sudo apt-get install libta-lib-dev", Colors.YELLOW)
                print_colored("2. Then run: pip install ta-lib", Colors.YELLOW)

def setup_directories():
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
        "config"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print_colored(f"✓ Created directory: {directory}", Colors.GREEN)
        else:
            print_colored(f"Directory already exists: {directory}", Colors.YELLOW)

def create_config_template():
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
            "demo": True
        },
        "default": {
            "coin_pair": "BTC/USDT",
            "leverage": 10,
            "timeframe": "1m",
            "retry_attempts": 3,
            "retry_delay_seconds": 5,
            "max_position_size": 0.01,
            "risk_per_trade": 0.02
        },
        "logging": {
            "level": "INFO",
            "max_file_size_mb": 10,
            "backup_count": 5
        },
        "performance": {
            "save_interval_minutes": 5,
            "backup_interval_hours": 24
        }
    }
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config_template, f, indent=2)
        print_colored("✓ Configuration template created at config/config.json", Colors.GREEN)
        print_colored("⚠ Please update the API keys before running the bot!", Colors.YELLOW)
    except Exception as e:
        print_colored(f"ERROR: Failed to create configuration template: {e}", Colors.RED)

def create_env_template():
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

# Trading Configuration
DEFAULT_COIN_PAIR=BTC/USDT
DEFAULT_LEVERAGE=10
DEFAULT_TIMEFRAME=1m

# Risk Management
MAX_POSITION_SIZE=0.01
RISK_PER_TRADE=0.02

# Logging
LOG_LEVEL=INFO
"""
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_template)
        print_colored("✓ Environment template created at .env.template", Colors.GREEN)
        print_colored("⚠ Copy to .env and update with your actual values!", Colors.YELLOW)
    except Exception as e:
        print_colored(f"ERROR: Failed to create environment template: {e}", Colors.RED)

def create_run_script():
    """Create platform-specific run scripts"""
    print_header("CREATING RUN SCRIPTS")
    
    # Windows batch script
    windows_script = """@echo off
echo Starting ByBit Trading Bot...
call venv\\Scripts\\activate
python bot.py
pause
"""
    
    # Linux/Mac shell script
    unix_script = """#!/bin/bash
echo "Starting ByBit Trading Bot..."
source venv/bin/activate
python bot.py
"""
    
    try:
        # Create Windows script
        with open("run_bot.bat", 'w') as f:
            f.write(windows_script)
        print_colored("✓ Windows run script created: run_bot.bat", Colors.GREEN)
        
        # Create Unix script
        with open("run_bot.sh", 'w') as f:
            f.write(unix_script)
        
        # Make Unix script executable
        if platform.system() != "Windows":
            os.chmod("run_bot.sh", 0o755)
        
        print_colored("✓ Unix run script created: run_bot.sh", Colors.GREEN)
        
    except Exception as e:
        print_colored(f"ERROR: Failed to create run scripts: {e}", Colors.RED)

def print_completion_message():
    """Print setup completion message"""
    print_header("SETUP COMPLETED SUCCESSFULLY!")
    
    print_colored("🎉 ByBit Trading Bot setup is complete!", Colors.GREEN)
    print_colored("\nNext steps:", Colors.BOLD)
    print_colored("1. Update config/config.json with your ByBit API credentials", Colors.YELLOW)
    print_colored("2. Copy .env.template to .env and configure your settings", Colors.YELLOW)
    print_colored("3. Run the bot using:", Colors.YELLOW)
    
    if platform.system() == "Windows":
        print_colored("   - Windows: run_bot.bat", Colors.BLUE)
        print_colored("   - Or: venv\\Scripts\\activate && python bot.py", Colors.BLUE)
    else:
        print_colored("   - Unix/Linux: ./run_bot.sh", Colors.BLUE)
        print_colored("   - Or: source venv/bin/activate && python bot.py", Colors.BLUE)
    
    print_colored("\n⚠ IMPORTANT SECURITY NOTES:", Colors.RED)
    print_colored("- Never commit your API keys to version control", Colors.RED)
    print_colored("- Start with testnet/demo mode before live trading", Colors.RED)
    print_colored("- Test thoroughly with small amounts first", Colors.RED)
    
    print_colored(f"\n{'='*60}", Colors.BLUE)

def main():
    """Main setup function"""
    print_header("BYBIT TRADING BOT SETUP")
    print_colored("Setting up your cryptocurrency trading bot environment...", Colors.BLUE)
    
    try:
        check_python_version()
        check_system_requirements()
        create_virtual_environment()
        install_dependencies()
        setup_directories()
        create_config_template()
        create_env_template()
        create_run_script()
        print_completion_message()
        
    except KeyboardInterrupt:
        print_colored("\n\nSetup cancelled by user", Colors.YELLOW)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\nUnexpected error during setup: {e}", Colors.RED)
        sys.exit(1)

if __name__ == "__main__":
    main() 