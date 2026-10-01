"""Persistent, single-writer upload supervision with a restart watchdog."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..io import atomic_json, file_hash


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress(folder: Path, state_path: Path) -> dict:
    manifest = json.loads((folder / "release_manifest.json").read_text())
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    uploaded = state.get("uploaded", {})
    manifest_path = folder / "release_manifest.json"
    files = {
        **manifest["files"],
        "release_manifest.json": {
            "bytes": manifest_path.stat().st_size,
            "sha256": file_hash(manifest_path),
        },
    }
    done = sum(v["bytes"] for k, v in files.items() if uploaded.get(k) == v["sha256"])
    total = sum(v["bytes"] for v in files.values())
    return {
        "committed_bytes": done,
        "total_bytes": total,
        "committed_percent": round(100 * done / total, 3) if total else 100,
        "committed_files": sum(uploaded.get(k) == v["sha256"] for k, v in files.items()),
        "total_files": len(files),
    }


def terminate_worker(worker) -> None:
    if worker.poll() is None:
        os.killpg(worker.pid, signal.SIGTERM)
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait()


def supervise(config_path: Path, key: str, token_file: Path | None = None) -> dict:
    """Remain alive until success or an explicit signal; never reset completed batches."""
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    entry = next(e for e in config["repositories"] if e["key"] == key)
    scratch = Path(config["scratch_root"]).resolve() / key
    scratch.mkdir(parents=True, exist_ok=True)
    folder = Path(config["staging_root"]).resolve() / key
    state_path = scratch / "upload_state.json"
    status_path = scratch / "supervisor_status.json"
    settings = config.get("upload_policy", {})
    watchdog = float(settings.get("watchdog_seconds", 3600))
    retry_base = float(settings.get("restart_delay_seconds", 30))
    retry_max = float(settings.get("max_restart_delay_seconds", 300))
    blocked_delay = float(settings.get("blocked_retry_seconds", 900))
    if min(watchdog, retry_base, retry_max, blocked_delay) <= 0:
        raise ValueError("Supervisor intervals must be positive")
    stopping = False
    worker = None

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (scratch / "supervisor.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another upload supervisor owns this repository") from None
            attempt = 0
            while not stopping:
                attempt += 1
                command = [
                    sys.executable,
                    "-m",
                    "pct4b.hub.cli",
                    "upload",
                    "--config",
                    str(config_path),
                    "--only",
                    key,
                    "--execute",
                ]
                if token_file:
                    command += ["--token-file", str(token_file.resolve())]
                started = time.monotonic()
                last_progress = started
                last_bytes = -1
                timed_out = False
                log_path = scratch / f"attempt-{attempt}-{time.time_ns()}.log"
                with log_path.open("w") as log:
                    worker = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    )
                    while worker.poll() is None and not stopping:
                        measured = progress(folder, state_path)
                        if measured["committed_bytes"] != last_bytes:
                            last_bytes = measured["committed_bytes"]
                            last_progress = time.monotonic()
                        status = {
                            "state": "uploading",
                            "repo_id": entry["repo_id"],
                            "supervisor_pid": os.getpid(),
                            "worker_pid": worker.pid,
                            "attempt": attempt,
                            "updated_at": timestamp(),
                            "seconds_without_commit": round(time.monotonic() - last_progress),
                            "attempt_log": log_path.name,
                            **measured,
                        }
                        atomic_json(status_path, status)
                        if time.monotonic() - last_progress > watchdog:
                            timed_out = True
                            terminate_worker(worker)
                            break
                        for _ in range(30):
                            if stopping or worker.poll() is not None:
                                break
                            time.sleep(1)
                    if stopping:
                        terminate_worker(worker)
                        break
                    code = worker.wait()
                if code == 0:
                    result = {
                        "state": "complete",
                        "repo_id": entry["repo_id"],
                        "updated_at": timestamp(),
                        **progress(folder, state_path),
                    }
                    atomic_json(status_path, result)
                    return result
                blocked = code == 78
                delay = (
                    blocked_delay
                    if blocked
                    else min(retry_max, retry_base * 2 ** min(attempt - 1, 8))
                )
                status = {
                    "state": "waiting_for_fix" if blocked else "retry_wait",
                    "repo_id": entry["repo_id"],
                    "supervisor_pid": os.getpid(),
                    "attempt": attempt,
                    "worker_exit_code": code,
                    "watchdog_restart": timed_out,
                    "updated_at": timestamp(),
                    "retry_after_seconds": delay,
                    **progress(folder, state_path),
                }
                atomic_json(status_path, status)
                print(json.dumps(status), flush=True)
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline and not stopping:
                    time.sleep(min(1, max(0, deadline - time.monotonic())))
            result = {
                "state": "stopped",
                "repo_id": entry["repo_id"],
                "updated_at": timestamp(),
                **progress(folder, state_path),
            }
            atomic_json(status_path, result)
            return result
    finally:
        if worker is not None:
            terminate_worker(worker)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
