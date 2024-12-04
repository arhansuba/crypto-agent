import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import json
import os
from enum import Enum

class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class LoggerSetup:
    def __init__(self, config: Dict):
        """Initialize logging system"""
        self.config = config
        self.log_dir = Path(config.get("log_dir", "logs"))
        self.log_dir.mkdir(exist_ok=True)
        
        # Create separate loggers for different components
        self.setup_main_logger()
        self.setup_trade_logger()
        self.setup_error_logger()

    def setup_main_logger(self) -> None:
        """Set up main application logger"""
        main_logger = logging.getLogger('main')
        main_logger.setLevel(self._get_log_level())
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_formatter())
        main_logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'app.log',
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(self._get_formatter())
        main_logger.addHandler(file_handler)

    def setup_trade_logger(self) -> None:
        """Set up trading activity logger"""
        trade_logger = logging.getLogger('trading')
        trade_logger.setLevel(logging.INFO)
        
        # JSON handler for trade logs
        trade_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'trades.json',
            maxBytes=10_000_000,
            backupCount=5
        )
        trade_handler.setFormatter(self._get_json_formatter())
        trade_logger.addHandler(trade_handler)

    def setup_error_logger(self) -> None:
        """Set up error logger"""
        error_logger = logging.getLogger('error')
        error_logger.setLevel(logging.ERROR)
        
        # Error file handler
        error_handler = logging.handlers.TimedRotatingFileHandler(
            self.log_dir / 'errors.log',
            when='midnight',
            interval=1,
            backupCount=30
        )
        error_handler.setFormatter(self._get_error_formatter())
        error_logger.addHandler(error_handler)
        
        # Critical error notifications
        if self.config.get("error_notifications", False):
            email_handler = self._get_email_handler()
            email_handler.setLevel(logging.CRITICAL)
            error_logger.addHandler(email_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get logger by name"""
        return logging.getLogger(name)

    def _get_formatter(self) -> logging.Formatter:
        """Get standard log formatter"""
        return logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _get_error_formatter(self) -> logging.Formatter:
        """Get error log formatter"""
        return logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d]\n'
            'Message: %(message)s\n'
            'Exception: %(exc_info)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    class JSONFormatter(logging.Formatter):
        """JSON formatter for structured logging"""
        def format(self, record):
            log_data = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage()
            }
            
            if hasattr(record, "trade_data"):
                log_data.update(record.trade_data)
            
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
                
            return json.dumps(log_data)

    def _get_json_formatter(self) -> 'JSONFormatter':
        """Get JSON formatter"""
        return self.JSONFormatter()

    def _get_email_handler(self) -> logging.handlers.SMTPHandler:
        """Get email handler for critical errors"""
        email_config = self.config.get("email", {})
        return logging.handlers.SMTPHandler(
            mailhost=email_config.get("smtp_host", "localhost"),
            fromaddr=email_config.get("from_addr", "bot@example.com"),
            toaddrs=email_config.get("to_addrs", ["admin@example.com"]),
            subject="Trading Bot Critical Error"
        )

    def _get_log_level(self) -> int:
        """Get configured log level"""
        level = self.config.get("log_level", "INFO").upper()
        return getattr(logging, level, logging.INFO)

class TradeLogger:
    """Helper class for logging trading activities"""
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_trade(self, trade_data: Dict) -> None:
        """Log trade information"""
        log_record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Trade executed",
            args=(),
            exc_info=None
        )
        log_record.trade_data = trade_data
        self.logger.handle(log_record)

    def log_analysis(self, analysis_data: Dict) -> None:
        """Log analysis information"""
        log_record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Analysis completed",
            args=(),
            exc_info=None
        )
        log_record.trade_data = analysis_data
        self.logger.handle(log_record)

def get_logger(name: str) -> logging.Logger:
    """Convenience function to get logger"""
    return logging.getLogger(name)

def setup_logging(config: Dict) -> LoggerSetup:
    """Set up logging system"""
    return LoggerSetup(config)