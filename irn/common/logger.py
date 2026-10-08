"""Structured logging system with reason codes and standardized formatting."""

import logging
import sys
from typing import Optional

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


class ReasonCodeFormatter(logging.Formatter):
    """Formats logs with timestamp, module, reason code, and message."""

    def format(self, record: logging.LogRecord) -> str:
        reason_code = getattr(record, "reason_code", "GEN-000")
        color = ""
        if record.levelno >= logging.ERROR:
            color = RED
        elif record.levelno >= logging.WARNING:
            color = YELLOW
        elif record.levelno >= logging.INFO:
            color = CYAN

        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        prefix = f"{color}[{timestamp}] [{record.levelname:<7}] [{reason_code}]{RESET}"
        return f"{prefix} {record.getMessage()}"


def setup_logger(name: str = "irn", level: int = logging.INFO) -> logging.Logger:
    """Configure and return the root IRN logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(ReasonCodeFormatter())
        logger.addHandler(handler)

    return logger


logger = setup_logger("irn")


def log_event(
    msg: str,
    reason_code: str = "INFO-001",
    level: int = logging.INFO,
    **kwargs
) -> None:
    """Helper to log structured events with explicit reason codes."""
    logger.log(level, msg, extra={"reason_code": reason_code}, **kwargs)
