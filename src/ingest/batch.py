from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set


@dataclass
class BatchItemResult:
    filename: str
    status: str  # created | overwritten | skipped | failed
    chunks_created: int = 0
    deleted_chunks: int = 0
    error: Optional[str] = None


def run_batch_ingest(
    *,
    items: List[Dict],
    existing_filenames: Set[str],
    overwrite_names: Set[str],
    save_file: Callable[[Dict], str],
    ingest_fn: Callable[..., Dict],
    on_progress: Optional[Callable[[int, int, str, str], None]] = None,
) -> List[BatchItemResult]:
    results: List[BatchItemResult] = []
    total = len(items)
    for i, item in enumerate(items, start=1):
        filename = item["filename"]
        if filename in existing_filenames and filename not in overwrite_names:
            if on_progress:
                on_progress(i, total, filename, "skipped")
            results.append(BatchItemResult(filename=filename, status="skipped"))
            continue
        if on_progress:
            on_progress(i, total, filename, "start")
        file_path = None
        try:
            file_path = save_file(item)
            result = ingest_fn(file_path, filename, True)
            status = "overwritten" if result.get("overwritten") else "created"
            results.append(
                BatchItemResult(
                    filename=filename,
                    status=status,
                    chunks_created=int(result.get("chunks_created", 0)),
                    deleted_chunks=int(result.get("deleted_chunks", 0)),
                )
            )
        except Exception as exc:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            results.append(
                BatchItemResult(filename=filename, status="failed", error=str(exc))
            )
        if on_progress:
            on_progress(i, total, filename, "done")
    return results
