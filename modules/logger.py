import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
import colorama
from colorama import Fore, Style
import re
import platform

# Initialize colorama for Windows
colorama.init()

LOG_DIR = 'logs'
LOG_FILE = 'bot.log'
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

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

def get_logger(name: str = 'bot') -> logging.Logger:
    logger = logging.getLogger(name)
    # Only configure if no handlers (per logger)
    if not logger.hasHandlers():
        logger.setLevel(LOG_LEVEL)
        
        # File handler with no colors
        file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        
        # Use different rotation strategy based on platform
        if platform.system() == 'Windows':
            # Use time-based rotation on Windows to avoid file handle issues
            fh = TimedRotatingFileHandler(
                LOG_PATH, 
                when='midnight', 
                interval=1, 
                backupCount=BACKUP_COUNT, 
                encoding='utf-8'
            )
            # Set suffix after initialization
            fh.suffix = '%Y-%m-%d'
        else:
            # Use size-based rotation on Unix systems
            fh = RotatingFileHandler(LOG_PATH, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8')
        
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
    return logger 