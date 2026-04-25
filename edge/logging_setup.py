"""
Environment-aware logging configuration.

Development: verbose — keep all loggers at the configured level.
Production: quiet — silence known-noisy third-party libraries to WARNING
            and disable uvicorn's per-request access log.

Call configure_logging() exactly once, early in the app lifecycle
(before any logger.info() is emitted) to take effect.
"""

import logging

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "apscheduler",
    "apscheduler.scheduler",
    "apscheduler.executors.default",
    "uvicorn.access",
    "supervision",
    "ultralytics",
    "PIL",
    "matplotlib",
)


def configure_logging(log_level: str = "INFO",
                      environment: str = "development") -> None:
    """
    Configure the root logger and silence noisy third-party loggers in prod.

    Args:
        log_level: minimum level for the root logger (DEBUG/INFO/WARNING/...)
        environment: "development" or "production"
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.getLogger("edge").setLevel(level)

    if environment == "production":
        # Silence noisy third-party libraries.
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

        logging.getLogger("uvicorn.access").disabled = True
