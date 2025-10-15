"""
Structured logging system for Symptom Recommendation API

Provides centralized logging configuration with file rotation
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(
    name: str = "symptom_api",
    log_file: str = "symptom_api.log",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup structured logger with file rotation

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # File handler with rotation
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "symptom_api") -> logging.Logger:
    """
    Get existing logger or create new one

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger not configured, set it up
    if not logger.handlers:
        return setup_logger(name)

    return logger


class LoggerAdapter:
    """
    Adapter for adding contextual information to logs
    """

    def __init__(self, logger: logging.Logger, context: dict = None):
        self.logger = logger
        self.context = context or {}

    def _add_context(self, message: str) -> str:
        """Add context information to log message"""
        if self.context:
            context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{message} | {context_str}"
        return message

    def debug(self, message: str):
        self.logger.debug(self._add_context(message))

    def info(self, message: str):
        self.logger.info(self._add_context(message))

    def warning(self, message: str):
        self.logger.warning(self._add_context(message))

    def error(self, message: str):
        self.logger.error(self._add_context(message))

    def critical(self, message: str):
        self.logger.critical(self._add_context(message))


# Example usage
if __name__ == "__main__":
    # Setup logger
    logger = setup_logger("test", "test.log", level=logging.DEBUG)

    # Basic logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # With context
    adapter = LoggerAdapter(logger, {"user_id": "12345", "request_id": "abc-123"})
    adapter.info("Processing recommendation request")
    adapter.error("Failed to load model")

    print("\nLog file created: test.log")
