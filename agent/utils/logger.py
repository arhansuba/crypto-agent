"""
Advanced logging system for AI crypto agent.
Provides structured logging with multiple handlers and custom formatters.
"""

import logging
import logging.handlers
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime, timedelta
import sys
import traceback
import queue
from concurrent.futures import ThreadPoolExecutor
import atexit
import os

class CryptoAgentLogger:
    """
    Advanced logging system with support for multiple log levels,
    rotating files, and asynchronous logging.
    """
    
    def __init__(self, config: Dict):
        """Initialize the logging system"""
        self.config = config
        self.log_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Create log directory
        self.log_dir = Path(config['logging']['directory'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self.main_logger = self._setup_main_logger()
        self.trade_logger = self._setup_trade_logger()
        self.error_logger = self._setup_error_logger()
        
        # Start async logging
        self._start_async_logging()
        
        # Register cleanup
        atexit.register(self.cleanup)

    def _setup_main_logger(self) -> logging.Logger:
        """Setup main application logger"""
        logger = logging.getLogger('CryptoAgent')
        logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.config['logging']['console_level'])
        console_handler.setFormatter(self._get_console_formatter())
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'crypto_agent.log',
            maxBytes=self.config['logging']['max_file_size'],
            backupCount=self.config['logging']['backup_count']
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self._get_file_formatter())
        logger.addHandler(file_handler)
        
        return logger

    def _setup_trade_logger(self) -> logging.Logger:
        """Setup trade-specific logger"""
        logger = logging.getLogger('CryptoAgent.Trades')
        logger.setLevel(logging.INFO)
        
        # Trade file handler
        trade_handler = logging.handlers.TimedRotatingFileHandler(
            self.log_dir / 'trades.log',
            when='midnight',
            interval=1,
            backupCount=self.config['logging']['trade_backup_count']
        )
        trade_handler.setFormatter(self._get_trade_formatter())
        logger.addHandler(trade_handler)
        
        return logger

    def _setup_error_logger(self) -> logging.Logger:
        """Setup error-specific logger"""
        logger = logging.getLogger('CryptoAgent.Errors')
        logger.setLevel(logging.ERROR)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'errors.log',
            maxBytes=self.config['logging']['max_file_size'],
            backupCount=self.config['logging']['backup_count']
        )
        error_handler.setFormatter(self._get_error_formatter())
        logger.addHandler(error_handler)
        
        # Email handler for critical errors
        if self.config['logging'].get('email_notifications'):
            email_handler = logging.handlers.SMTPHandler(
                mailhost=self.config['logging']['smtp_server'],
                fromaddr=self.config['logging']['email_from'],
                toaddrs=self.config['logging']['email_to'],
                subject='CryptoAgent Critical Error'
            )
            email_handler.setLevel(logging.CRITICAL)
            email_handler.setFormatter(self._get_error_formatter())
            logger.addHandler(email_handler)
        
        return logger

    def _get_console_formatter(self) -> logging.Formatter:
        """Create console log formatter"""
        return logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _get_file_formatter(self) -> logging.Formatter:
        """Create file log formatter"""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _get_trade_formatter(self) -> logging.Formatter:
        """Create trade log formatter"""
        class TradeFormatter(logging.Formatter):
            def format(self, record):
                if hasattr(record, 'trade_data'):
                    # Format trade data as JSON
                    record.msg = json.dumps(record.trade_data)
                return super().format(record)
        
        return TradeFormatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _get_error_formatter(self) -> logging.Formatter:
        """Create error log formatter"""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d\n'
            'Message: %(message)s\n'
            'Exception: %(exc_info)s\n'
            'Stack Trace:\n%(stack_info)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _start_async_logging(self):
        """Start asynchronous logging process"""
        def log_worker():
            while True:
                try:
                    record = self.log_queue.get()
                    if record is None:
                        break
                    logger = logging.getLogger(record.name)
                    logger.handle(record)
                except Exception:
                    traceback.print_exc()
        
        self.executor.submit(log_worker)

    def cleanup(self):
        """Cleanup logging resources"""
        self.log_queue.put(None)
        self.executor.shutdown()

    def log_trade(
        self,
        trade_data: Dict,
        level: int = logging.INFO
    ):
        """Log trade information"""
        record = logging.LogRecord(
            name=self.trade_logger.name,
            level=level,
            pathname='',
            lineno=0,
            msg='',
            args=(),
            exc_info=None
        )
        record.trade_data = trade_data
        self.log_queue.put(record)

    def log_error(
        self,
        error: Union[str, Exception],
        level: int = logging.ERROR
    ):
        """Log error information"""
        if isinstance(error, Exception):
            exc_info = (type(error), error, error.__traceback__)
            message = str(error)
        else:
            exc_info = sys.exc_info()
            message = error
        
        record = logging.LogRecord(
            name=self.error_logger.name,
            level=level,
            pathname=traceback.extract_stack()[-2].filename,
            lineno=traceback.extract_stack()[-2].lineno,
            msg=message,
            args=(),
            exc_info=exc_info
        )
        record.stack_info = ''.join(traceback.format_stack())
        self.log_queue.put(record)

    def log_market_event(
        self,
        event_type: str,
        event_data: Dict,
        level: int = logging.INFO
    ):
        """Log market-related events"""
        message = f"Market Event: {event_type} - {json.dumps(event_data)}"
        record = logging.LogRecord(
            name=self.main_logger.name,
            level=level,
            pathname='',
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        self.log_queue.put(record)

    def log_prediction(
        self,
        prediction_data: Dict,
        level: int = logging.INFO
    ):
        """Log model predictions"""
        message = f"Model Prediction: {json.dumps(prediction_data)}"
        record = logging.LogRecord(
            name=self.main_logger.name,
            level=level,
            pathname='',
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        self.log_queue.put(record)

    def get_recent_trades(
        self,
        hours: int = 24,
        status: Optional[str] = None
    ) -> List[Dict]:
        """Retrieve recent trade logs"""
        trades = []
        log_file = self.log_dir / 'trades.log'
        
        if not log_file.exists():
            return trades
        
        start_time = datetime.now() - timedelta(hours=hours)
        
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    timestamp_str = line.split(' - ')[0]
                    timestamp = datetime.strptime(
                        timestamp_str,
                        '%Y-%m-%d %H:%M:%S'
                    )
                    
                    if timestamp < start_time:
                        continue
                    
                    trade_data = json.loads(line.split(' - ')[1])
                    
                    if status and trade_data.get('status') != status:
                        continue
                    
                    trades.append(trade_data)
                    
                except Exception:
                    continue
        
        return trades

    def get_error_summary(
        self,
        hours: int = 24,
        min_level: int = logging.ERROR
    ) -> List[Dict]:
        """Retrieve recent error logs"""
        errors = []
        log_file = self.log_dir / 'errors.log'
        
        if not log_file.exists():
            return errors
        
        start_time = datetime.now() - timedelta(hours=hours)
        
        with open(log_file, 'r') as f:
            current_error = {}
            for line in f:
                try:
                    if line.startswith('20'):  # New error entry
                        if current_error:
                            errors.append(current_error)
                        
                        parts = line.split(' - ')
                        timestamp = datetime.strptime(
                            parts[0],
                            '%Y-%m-%d %H:%M:%S'
                        )
                        
                        if timestamp < start_time:
                            current_error = {}
                            continue
                        
                        current_error = {
                            'timestamp': timestamp,
                            'level': parts[2],
                            'location': parts[3],
                            'message': '',
                            'stack_trace': ''
                        }
                    elif current_error:
                        if line.startswith('Message:'):
                            current_error['message'] = line[9:].strip()
                        elif line.startswith('Stack Trace:'):
                            current_error['stack_trace'] = line[12:].strip()
                    
                except Exception:
                    continue
            
            if current_error:
                errors.append(current_error)
        
        return [e for e in errors if logging.getLevelName(e['level']) >= min_level]