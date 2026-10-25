"""Central logging setup so every module logs in one consistent format."""

import logging

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format=_FORMAT, force=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"research_assistant.{name}")
