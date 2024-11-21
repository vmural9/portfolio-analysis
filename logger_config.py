import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logger():
    # Create logger
    logger = logging.getLogger('portfolio_analyzer')
    logger.setLevel(logging.DEBUG)

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')

    # Create handlers
    # File handler with rotation (max 5MB per file, keep 3 backup files)
    file_handler = RotatingFileHandler('portfolio_analyzer.log', maxBytes=5*1024*1024, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger 