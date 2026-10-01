"""Parallel blob transfers with one commit writer and per-file retries."""

from __future__ import annotations

import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone

from ..io import atomic_json


def upload_parallel(api, entry, folder, pending, hashes, state, state_path, workers, batch_size):
    from huggingface_hub import CommitOperationAdd

    from .publish import RetryableUploadError, is_transient_upload_error

    transferred, ready, failed = [], [], []
    status_path = state_path.with_name("transfer_status.json")
    started = time.monotonic()
    initial_bytes = sum(
        (folder / n).stat().st_size for n in hashes if state["uploaded"].get(n) == hashes[n]
    )
    transferred_bytes = 0

    def transfer(name):
        op = CommitOperationAdd(path_in_repo=name, path_or_fileobj=folder / name)
        for attempt in range(4):
            try:
                api.preupload_lfs_files(
                    entry["repo_id"], [op], repo_type=entry["repo_type"], num_threads=1
                )
                return name, op
            except Exception as exc:
                if not is_transient_upload_error(exc) or attempt == 3:
                    raise
                print(
                    json.dumps(
                        {
                            "retry_file": name,
                            "attempt": attempt + 1,
                            "error_type": type(exc).__name__,
                        }
                    ),
                    flush=True,
                )
                time.sleep(min(30, 2**attempt))

    def commit_ready():
        if not ready:
            return
        batch = ready[:batch_size]
        for attempt in range(4):
            try:
                result = api.create_commit(
                    repo_id=entry["repo_id"],
                    repo_type=entry["repo_type"],
                    operations=[op for _, op in batch],
                    commit_message=f"Release {entry['title']} ({len(batch)} verified files)",
                    num_threads=workers,
                )
                break
            except Exception as exc:
                if not is_transient_upload_error(exc) or attempt == 3:
                    raise
                time.sleep(min(30, 2**attempt))
        state["uploaded"].update({name: hashes[name] for name, _ in batch})
        atomic_json(state_path, state)
        del ready[: len(batch)]
        print(
            json.dumps(
                {"uploaded": entry["repo_id"], "batch_files": len(batch), "commit": result.oid}
            ),
            flush=True,
        )

    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        active = {}

        def replenish():
            while len(active) < workers:
                name = next(iterator, None)
                if name is None:
                    break
                active[executor.submit(transfer, name)] = name

        replenish()
        while active:
            completed, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in completed:
                name = active.pop(future)
                try:
                    result = future.result()
                    ready.append(result)
                    transferred.append(name)
                    transferred_bytes += (folder / name).stat().st_size
                except Exception as exc:
                    # Healthy transfers continue; the supervisor retries failed files later.
                    failed.append((name, is_transient_upload_error(exc), type(exc).__name__))
                    print(
                        json.dumps({"deferred_file": name, "error_type": type(exc).__name__}),
                        flush=True,
                    )
            replenish()
            if len(ready) >= batch_size or (not completed and ready) or not active:
                commit_ready()
            seconds = max(0.001, time.monotonic() - started)
            atomic_json(
                status_path,
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "mode": "parallel_http",
                    "workers": workers,
                    "active_files": len(active),
                    "preuploaded_files_this_run": len(transferred),
                    "preuploaded_bytes_this_run": transferred_bytes,
                    "initial_committed_bytes": initial_bytes,
                    "elapsed_seconds": round(seconds, 1),
                    "effective_MB_per_second": round(transferred_bytes / seconds / 1e6, 3),
                    "deferred_files": len(failed),
                },
            )
        while ready:
            commit_ready()
    if failed:
        error = (
            RetryableUploadError if all(temporary for _, temporary, _ in failed) else RuntimeError
        )
        raise error(f"{len(failed)} files remain; successful commits were preserved")
