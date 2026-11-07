"""Process-pool helper shared by the stages that run RDKit per molecule."""

import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

CHUNK = 2000
# Always "spawn": by the time a pool starts, Polars has started its own threads,
# and fork() of a multi-threaded process can deadlock the child (Linux default).
_CONTEXT = multiprocessing.get_context("spawn")


def _silence_stdout() -> None:
    """Send this process's C-level stdout to /dev/null.

    RDKit's extended-Hückel backend prints a warning for every short X-H bond
    (O-H ~0.96 Å is "suspicious" to it), tens of thousands of lines per run.
    Used as the worker initializer, so the main process keeps its output.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def map_chunks(
    fn: Callable[[list[tuple[Any, ...]]], list[dict[str, Any]]],
    rows: list[tuple[Any, ...]],
    workers: int,
) -> list[dict[str, Any]]:
    """Apply ``fn`` to fixed-size chunks of ``rows`` in worker processes, keeping order."""
    chunks = [rows[i : i + CHUNK] for i in range(0, len(rows), CHUNK)]
    with ProcessPoolExecutor(
        max_workers=max(1, workers), mp_context=_CONTEXT, initializer=_silence_stdout
    ) as pool:
        return [r for chunk in pool.map(fn, chunks) for r in chunk]
