import logging
import os
from logging.handlers import RotatingFileHandler
import json

LOG_DIR = 'logs'
LOG_FILE = 'bot.log'
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

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
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        # Rotating file handler
        fh = RotatingFileHandler(LOG_PATH, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8')
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(LOG_LEVEL)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # Prevent double-logging through root logger
        logger.propagate = False
    return logger 