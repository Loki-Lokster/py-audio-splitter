import os
import sys
import logging
from datetime import datetime
from .utils import print_status

def setup_logging():
    """Setup logging configuration with log rotation"""
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
        
        # Remove excess log files
        for old_log in log_files[MAX_LOGS-1:]:
            try:
                os.remove(os.path.join(log_dir, old_log))
                logging.info(f"Removed old log file: {old_log}")
            except OSError:
                pass
    except Exception as e:
        print_status(f"Error cleaning up logs: {str(e)}", "warning")
    
    # Create new log file
    log_file = os.path.join(log_dir, f'audio_splitter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Log system info at startup
    logging.info("=== Audio Splitter Started ===")
    logging.info(f"Python Version: {sys.version}")
    logging.info(f"Platform: {sys.platform}")
    
    # Log cleanup summary
    if os.path.exists(log_dir):
        current_logs = len([f for f in os.listdir(log_dir) if f.endswith('.log')])
        logging.info(f"Maintaining {current_logs} most recent log files in {log_dir}")
    
    return log_file 