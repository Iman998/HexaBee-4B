"""Streaming records, atomic output, and content-addressed provenance."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from itertools import islice
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_records(path, limit=None):
    """Auto-detect JSON arrays and JSONL/concatenated JSON objects by content.

    The original Wikipedia file ends in .json but contains JSONL. Pandas column-
    oriented JSON must be exported to records/JSONL before downstream validation.
    """
    import ijson

    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(4096)
            if not chunk:
                return
            lead = chunk.lstrip()
            if lead:
                break
        handle.seek(0)
        prefix = "item" if lead.startswith(b"[") else ""
        iterator = ijson.items(handle, prefix, multiple_values=not bool(prefix), use_float=True)
        for row in islice(iterator, limit):
            if not isinstance(row, dict):
                raise ValueError("Expected JSON records, not scalars or nested arrays")
            yield row


def atomic_json(path, value):
    return atomic_write(
        path, lambda f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    )


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            result = writer(handle)
            handle.write("\n")
        os.replace(name, path)
        return result
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_records(path, rows):
    def emit(handle):
        n = 0
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            n += 1
        return n

    return atomic_write(path, emit)


def manifest(path, command, inputs, settings, counts):
    """Only basenames and hashes are published; credentials and endpoints are excluded."""
    atomic_json(
        str(path) + ".manifest.json",
        {
            "schema_version": 1,
            "command": command,
            "inputs": [{"name": Path(p).name, "sha256": file_hash(p)} for p in inputs],
            "settings": settings,
            "counts": counts,
            "output_sha256": file_hash(path),
        },
    )
