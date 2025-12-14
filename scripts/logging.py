import os
import sys
import logging
from datetime import datetime
from .utils import print_status, Colors

def setup_logging(log_dir=None, console: bool = True):
    """Setup logging configuration with log rotation"""
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Keep only the last 5 log files
    MAX_LOGS = 5
    
    # Clean up old logs
    try:
        log_files = sorted([
            f for f in os.listdir(log_dir) 
            if f.startswith('audio_splitter_') and f.endswith('.log')
        ], reverse=True)
        
        # Remove excess log files (leave room for the new log file)
        max_old_logs = max(0, MAX_LOGS - 1)
        for old_log in log_files[max_old_logs:]:
            try:
                os.remove(os.path.join(log_dir, old_log))
            except OSError:
                pass
    except Exception as e:
        print_status(f"Error cleaning up logs: {str(e)}", "warning")
    
    # Create new log file
    log_file = os.path.join(log_dir, f'audio_splitter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            # Add colors to log levels
            if record.levelno == logging.WARNING:
                record.msg = f"{Colors.RED}{record.msg}{Colors.RESET}"
            elif record.levelno == logging.ERROR:
                record.msg = f"{Colors.RED}{record.msg}{Colors.RESET}"
            elif record.levelno == logging.INFO:
                # Don't color info messages to keep them clean
                record.msg = f"{record.msg}"
            
            return super().format(record)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console_formatter = ColoredFormatter('%(message)s')  # Simpler format for console
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    root_logger.handlers = []
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Log system info at startup
    logging.info("=== Audio Splitter Started ===")
    logging.info(f"Python Version: {sys.version}")
    logging.info(f"Platform: {sys.platform}")
    
    # Log cleanup summary
    if os.path.exists(log_dir):
        current_logs = len([f for f in os.listdir(log_dir) if f.endswith('.log')])
        logging.info(f"Maintaining {current_logs} most recent log files in {log_dir}")
    
    return log_file 
