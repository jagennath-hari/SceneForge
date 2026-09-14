"""Terminal progress with elapsed-time refresh during blocking operations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
import logging
import sys
from threading import Event, Thread
from time import monotonic
from typing import TypeVar

from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class Progress:
    """Count completed work only; a heartbeat is not evidence of GPU progress."""

    def __init__(self, description: str, total: int | None = None, unit: str = "item") -> None:
        self.description = description
        self.detail = "working"
        self.interactive = sys.stderr.isatty()
        self.bar = tqdm(total=total, desc=description, unit=unit, dynamic_ncols=True,
                        mininterval=0.25, disable=not self.interactive,
                        bar_format="{desc}: {elapsed} | {postfix}" if total is None and unit == "item" else None)
        self.completed = 0
        self.started = monotonic()
        self.stopped = Event()
        self.thread = Thread(target=self._refresh, name="progress-refresh", daemon=True)

    def __enter__(self) -> Progress:
        if not self.interactive:
            LOGGER.info("%s: started", self.description)
        self.thread.start()
        return self

    def status(self, detail: str) -> None:
        self.detail = detail
        self.bar.set_postfix_str(detail, refresh=False)

    def advance(self, count: int = 1) -> None:
        self.completed += count
        self.bar.update(count)

    def add_work(self, count: int = 1) -> None:
        if self.bar.total is not None:
            self.bar.total += count

    def _refresh(self) -> None:
        interval = 1 if self.interactive else 15
        while not self.stopped.wait(interval):
            if self.interactive:
                self.bar.refresh()
            else:
                LOGGER.info("%s: %d completed, %.0fs elapsed — %s",
                            self.description, self.completed, monotonic() - self.started, self.detail)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stopped.set()
        self.thread.join()
        self.status("done" if exc_type is None else "stopped")
        self.bar.close()
        if not self.interactive:
            LOGGER.info("%s: %s, %d completed in %.1fs", self.description, self.detail,
                        self.completed, monotonic() - self.started)


@contextmanager
def tracked(items: Iterable[T], description: str, total: int, unit: str = "item") -> Iterator[Iterator[T]]:
    """Advance after each loop body; close the bar even when that body fails."""
    with Progress(description, total, unit) as progress:
        def iterate() -> Iterator[T]:
            for item in items:
                yield item
                progress.advance()
        yield iterate()
