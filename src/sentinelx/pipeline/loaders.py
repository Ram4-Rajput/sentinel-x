"""Streaming / chunked loaders for large flow CSV / .binetflow files.

Design:
  * True streaming via the stdlib ``csv`` module (row-by-row); memory stays
    O(chunk_size), never O(file). No full-file load.
  * Chunked generator interface: ``iter_raw_chunks`` yields lists of dict rows;
    ``iter_unified_records`` yields adapter-mapped UnifiedFlowRecords.
  * Malformed rows (wrong field count, unparseable) are counted + skipped, not
    fatal. Progress is logged every ``log_every`` rows.
  * Schema validation runs once on the header via the existing adapter.
  * Header-less UNSW raw files: the caller passes ``headerless=True`` and the
    loader attaches the canonical 49 names before mapping.
  * Routing: the dataset key selects the existing adapter via get_adapter().

Nothing here re-implements mapping — it delegates to sentinelx.data adapters.
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence

from ..data.adapters import get_adapter
from ..data.adapters.base import SchemaValidationError
from ..data.adapters.unsw_nb15 import RAW_COLUMN_ORDER, attach_raw_header
from ..data.schema import UnifiedFlowRecord

# Allow very large CSV fields (some flow exports have long cells).
try:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
except OverflowError:  # pragma: no cover - platform dependent
    csv.field_size_limit(2**31 - 1)


@dataclass
class LoaderStats:
    files_processed: int = 0
    rows_read: int = 0
    records_yielded: int = 0
    malformed_rows: int = 0
    skipped_records: int = 0
    files: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "files_processed": self.files_processed,
            "rows_read": self.rows_read,
            "records_yielded": self.records_yielded,
            "malformed_rows": self.malformed_rows,
            "skipped_records": self.skipped_records,
            "files": list(self.files),
        }


def _default_logger(msg: str) -> None:
    print(msg, flush=True)


def iter_raw_chunks(
    files: Sequence[Path],
    *,
    chunk_size: int = 100_000,
    is_binetflow: bool = False,
    headerless: bool = False,
    stats: Optional[LoaderStats] = None,
    log: Optional[Callable[[str], None]] = None,
    log_every: int = 500_000,
) -> Iterator[List[Dict[str, object]]]:
    """Yield chunks (lists) of dict-rows streamed from the given files.

    For header-less files a synthetic header (RAW_COLUMN_ORDER) is applied.
    """
    log = log or _default_logger
    stats = stats if stats is not None else LoaderStats()
    buffer: List[Dict[str, object]] = []

    for path in files:
        stats.files.append(str(path))
        stats.files_processed += 1
        log(f"[loader] reading {path.name} ...")
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as fh:
            if headerless:
                reader = csv.reader(fh)
                header = list(RAW_COLUMN_ORDER)
                for values in reader:
                    stats.rows_read += 1
                    if len(values) != len(header):
                        stats.malformed_rows += 1
                        continue
                    buffer.append(dict(zip(header, values)))
                    if len(buffer) >= chunk_size:
                        yield buffer
                        buffer = []
                    if stats.rows_read % log_every == 0:
                        log(f"[loader]   {stats.rows_read:,} rows read")
            else:
                reader = csv.DictReader(fh)
                for row in reader:
                    stats.rows_read += 1
                    # csv.DictReader puts overflow fields under None key -> malformed
                    if None in row and row[None]:
                        stats.malformed_rows += 1
                        continue
                    buffer.append(row)
                    if len(buffer) >= chunk_size:
                        yield buffer
                        buffer = []
                    if stats.rows_read % log_every == 0:
                        log(f"[loader]   {stats.rows_read:,} rows read")
    if buffer:
        yield buffer


def _validate_header(dataset: str, files: Sequence[Path], headerless: bool) -> None:
    """Validate the first file's header against the adapter (once)."""
    if headerless or not files:
        return
    adapter = get_adapter(dataset)
    with open(files[0], "r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
    if header is None:
        raise SchemaValidationError(f"{dataset}: {files[0]} is empty (no header).")
    adapter.validate_schema(header)


def iter_unified_records(
    dataset: str,
    files: Sequence[Path],
    *,
    chunk_size: int = 100_000,
    is_binetflow: bool = False,
    headerless: bool = False,
    max_records: Optional[int] = None,
    stats: Optional[LoaderStats] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Iterator[UnifiedFlowRecord]:
    """Stream UnifiedFlowRecords for a dataset via its existing adapter.

    Memory-safe: processes one chunk at a time. ``max_records`` caps output for
    smoke/medium runs. Malformed / unmappable rows are skipped and counted.
    """
    log = log or _default_logger
    stats = stats if stats is not None else LoaderStats()
    adapter = get_adapter(dataset)

    _validate_header(dataset, files, headerless)

    produced = 0
    global_index = 0
    for chunk in iter_raw_chunks(
        files, chunk_size=chunk_size, is_binetflow=is_binetflow,
        headerless=headerless, stats=stats, log=log,
    ):
        for row in chunk:
            try:
                rec = adapter.map_row(row, global_index)
            except Exception:  # malformed cell values etc. -> skip, count
                stats.skipped_records += 1
                global_index += 1
                continue
            global_index += 1
            stats.records_yielded += 1
            produced += 1
            yield rec
            if max_records is not None and produced >= max_records:
                log(f"[loader] reached max_records={max_records:,}; stopping.")
                return
