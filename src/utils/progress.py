from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, TypeVar


T = TypeVar("T")
U = TypeVar("U")


def configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("CVIR_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def worker_count() -> int:
    configured = os.environ.get("CVIR_WORKERS")
    if configured:
        value = int(configured)
        if value < 1:
            raise ValueError("CVIR_WORKERS must be at least 1")
        return value
    return max(1, min(32, (os.cpu_count() or 1) + 4))


def parallel_map(
    function: Callable[[T], U],
    items: Iterable[T],
    description: str,
    log_every: int = 250,
) -> Iterator[U]:
    values = list(items)
    total = len(values)
    logger = logging.getLogger("cvir.progress")
    if total == 0:
        return

    logger.info("Starting %s: %d items with %d workers", description, total, worker_count())
    with ThreadPoolExecutor(max_workers=worker_count(), thread_name_prefix="cvir") as executor:
        for completed, result in enumerate(executor.map(function, values), start=1):
            if completed == 1 or completed == total or completed % log_every == 0:
                logger.info("%s: %d/%d complete", description, completed, total)
            yield result