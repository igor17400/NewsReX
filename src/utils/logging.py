import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration with rich handler.

    Args:
        level: Logging level (default: "INFO")
    """
    # Remove any existing handlers
    logging.root.handlers = []

    # Configure logging with rich handler
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True, markup=True, console=console, show_time=True, show_path=False, enable_link_path=False
            )
        ],
        force=True,  # Override any existing configuration
    )

    # Disable Hydra's logging
    logging.getLogger("hydra").setLevel(logging.WARNING)
    # Set other loggers to INFO
    logging.getLogger("tensorflow").setLevel(logging.INFO)

    # Disable propagation for some loggers
    logging.getLogger("hydra").propagate = False

    # Disable Hydra's logging
    logging.getLogger("hydra").setLevel(logging.WARNING)
    # Set other loggers to INFO
    logging.getLogger("tensorflow").setLevel(logging.INFO)
