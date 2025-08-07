import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
import colorama
from colorama import Fore, Style
import re
import platform
from datetime import datetime

# Initialize colorama for Windows
colorama.init()

LOG_DIR = 'logs'

# Global variables for dynamic configuration
_current_symbol = None
_current_date_str = None
_log_file_path = None
_configured_loggers = set()

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

def format_complex_message(msg):
    """Format complex messages (dicts, lists) into readable multi-line strings"""
    if isinstance(msg, (dict, list)):
        try:
            # Try to format as JSON with indentation
            formatted = json.dumps(msg, indent=2)
            # Add a newline before the formatted data
            return f"\n{formatted}"
        except:
            return str(msg)
    return str(msg)

# Custom formatter that adds colors
class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels"""
    
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        # Format complex messages
        record.msg = format_complex_message(record.msg)
        
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        
        # Format the message with colors
        if record.levelname == 'ERROR':
            record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
        elif record.levelname == 'WARNING':
            record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"
        
        return super().format(record)

# Base format without colors (for file logging)
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Colored format for console
CONSOLE_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

# Helper to load log level from config.json or environment
def _get_log_level():
    # 1. Check environment variable
    env_level = os.environ.get('LOG_LEVEL')
    if env_level:
        return getattr(logging, env_level.upper(), logging.INFO)
    # 2. Check config.json
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            log_level = config.get('log_level')
            if log_level:
                return getattr(logging, log_level.upper(), logging.INFO)
    except Exception:
        pass
    return logging.INFO

LOG_LEVEL = _get_log_level()

# Rotating file handler settings
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 5

def configure_logging_session(symbol: str, session_date: str = None) -> None:
    """
    Configure logging for a trading session with symbol and date information.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        session_date: Session date in YYYYMMDD format, defaults to today
    """
    global _current_symbol, _current_date_str, _log_file_path, _configured_loggers
    
    # Set current session info
    _current_symbol = symbol.replace('/', '').replace('USDT', '').upper()  # Clean symbol for filename
    _current_date_str = session_date or datetime.now().strftime('%Y%m%d')
    
    # Create log file path with symbol and date
    log_filename = f'bot_{_current_symbol}_{_current_date_str}.log'
    _log_file_path = os.path.join(LOG_DIR, log_filename)
    
    # Log the configuration
    print(f"📋 Logging configured for session: {_current_symbol} on {_current_date_str}")
    print(f"📄 Log file: {_log_file_path}")
    
    # Reconfigure existing loggers to use new file path
    _reconfigure_existing_loggers()

def _reconfigure_existing_loggers() -> None:
    """Reconfigure all existing loggers to use the new log file path"""
    global _configured_loggers
    
    if not _log_file_path:
        return
    
    # Get all existing loggers
    for logger_name in list(_configured_loggers):
        logger = logging.getLogger(logger_name)
        
        # Remove existing file handlers
        for handler in logger.handlers[:]:
            if isinstance(handler, (RotatingFileHandler, TimedRotatingFileHandler)):
                logger.removeHandler(handler)
                handler.close()
        
        # Add new file handler with updated path
        _add_file_handler(logger)

def _add_file_handler(logger: logging.Logger) -> None:
    """Add file handler to logger with current log file path"""
    if not _log_file_path:
        return
    
    # File handler with no colors
    file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    
    # Use different rotation strategy based on platform
    if platform.system() == 'Windows':
        # Use time-based rotation on Windows to avoid file handle issues
        fh = TimedRotatingFileHandler(
            _log_file_path, 
            when='midnight', 
            interval=1, 
            backupCount=BACKUP_COUNT, 
            encoding='utf-8'
        )
        # Set suffix after initialization
        fh.suffix = '%Y-%m-%d'
    else:
        # Use size-based rotation on Unix systems
        fh = RotatingFileHandler(_log_file_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8')
    
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

def get_session_info() -> dict:
    """Get current logging session information"""
    return {
        'symbol': _current_symbol,
        'date': _current_date_str,
        'log_file': _log_file_path
    }

def get_logger(name: str = 'bot') -> logging.Logger:
    """
    Get a logger with dynamic file naming based on current trading session.
    
    Args:
        name: Logger name (e.g., 'bot', 'strategy', 'risk_manager')
        
    Returns:
        Configured logger instance
    """
    global _configured_loggers
    
    logger = logging.getLogger(name)
    
    # Only configure if no handlers (per logger)
    if not logger.hasHandlers():
        logger.setLevel(LOG_LEVEL)
        
        # Add file handler if log path is configured
        if _log_file_path:
            _add_file_handler(logger)
        else:
            # Fallback to generic log file if session not configured yet
            fallback_path = os.path.join(LOG_DIR, f'bot_startup_{datetime.now().strftime("%Y%m%d")}.log')
            file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
            
            if platform.system() == 'Windows':
                fh = TimedRotatingFileHandler(
                    fallback_path, 
                    when='midnight', 
                    interval=1, 
                    backupCount=BACKUP_COUNT, 
                    encoding='utf-8'
                )
                fh.suffix = '%Y-%m-%d'
            else:
                fh = RotatingFileHandler(fallback_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8')
            
            fh.setLevel(LOG_LEVEL)
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)
        
        # Console handler with colors
        console_formatter = ColoredFormatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT)
        ch = logging.StreamHandler()
        ch.setLevel(LOG_LEVEL)
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)
        
        # Prevent double-logging through root logger
        logger.propagate = False
        
        # Track this logger for reconfiguration
        _configured_loggers.add(name)
    
    return logger

def close_all_loggers() -> None:
    """Close all logger handlers gracefully"""
    global _configured_loggers
    
    for logger_name in list(_configured_loggers):
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    
    _configured_loggers.clear() 