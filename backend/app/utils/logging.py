import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once.

    Request bodies, document text and credentials are never logged; settings
    hold the API key as ``SecretStr`` so it is masked even if printed.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    if not any(getattr(handler, "_researchmate", False) for handler in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler._researchmate = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    # Third-party libraries are chatty at INFO. uvicorn's access log duplicates
    # the request log in main.py, which also records timing and a request id.
    for noisy in (
        "uvicorn.access",
        "httpx",
        "httpx2",
        "httpcore",
        "urllib3",
        "sentence_transformers",
        "faiss",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
