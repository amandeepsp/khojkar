import logging
import sys

# ANSI color codes
COLORS = {
    "RESET": "\033[0m",
    "BLACK": "\033[30m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "BOLD_RED": "\033[1;31m",
}


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log level names using ANSI codes
    """

    LEVEL_COLORS = {
        logging.DEBUG: COLORS["CYAN"],
        logging.INFO: COLORS["GREEN"],
        logging.WARNING: COLORS["YELLOW"],
        logging.ERROR: COLORS["RED"],
        logging.CRITICAL: COLORS["BOLD_RED"],
    }

    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        color = self.LEVEL_COLORS.get(record.levelno, COLORS["RESET"])
        record.levelname = f"{color}{levelname}{COLORS['RESET']}"
        return super().format(record)


def configure_global_logging(level=logging.INFO):
    """
    Configure the root logger with compact colored format.
    After calling this once at application startup, all modules can use
    the standard logging module directly.

    Args:
        level: Logging level for the root logger
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers
    if root_logger.handlers:
        root_logger.handlers = []

    # Create a formatter with compact format
    formatter = ColoredFormatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
