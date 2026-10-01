"""Parallel upload regressions; no external requests."""

import json
import threading
import time
from concurrent.futures import wait as real_wait
from types import SimpleNamespace

import pytest

from pct4b.hub.parallel import upload_parallel
from pct4b.io import file_hash


def test_failed_transfer_preserves_other_commits(tmp_path, monkeypatch):
    import httpx

    from pct4b.hub.publish import RetryableUploadError

    monkeypatch.setattr("pct4b.hub.parallel.time.sleep", lambda _: None)
    files = ["bad.bin", "good1.bin", "good2.bin", "good3.bin"]
    for name in files:
        (tmp_path / name).write_bytes(name.encode())
    hashes = {n: file_hash(tmp_path / n) for n in files}
    state = {"uploaded": {}}
    threads = []
    committed = []

    class API:
        def preupload_lfs_files(self, repo, ops, **kwargs):
            if ops[0].path_in_repo == "bad.bin":
                raise httpx.ReadTimeout("temporary")
            ops[0]._is_uploaded = True

        def create_commit(self, operations, **kwargs):
            threads.append(threading.get_ident())
            assert all(op._is_uploaded for op in operations)
            committed.extend(op.path_in_repo for op in operations)
            return SimpleNamespace(oid="commit")

    with pytest.raises(RetryableUploadError):
        upload_parallel(
            API(),
            {"repo_id": "owner/data", "repo_type": "dataset", "title": "Data"},
            tmp_path,
            files,
            hashes,
            state,
            tmp_path / "upload_state.json",
            3,
            2,
        )
    assert set(committed) == set(files) - {"bad.bin"}
    assert set(json.loads((tmp_path / "upload_state.json").read_text())["uploaded"]) == set(
        committed
    )
    assert set(threads) == {threading.get_ident()}


def test_idle_poll_waits_to_fill_commit_batch(tmp_path, monkeypatch):
    from pct4b.hub import parallel

    for name in ("first.bin", "slow.bin"):
        (tmp_path / name).write_bytes(name.encode())
    files = ["first.bin", "slow.bin"]
    hashes = {name: file_hash(tmp_path / name) for name in files}
    commits = []

    class API:
        def preupload_lfs_files(self, repo, ops, **kwargs):
            if ops[0].path_in_repo == "slow.bin":
                time.sleep(0.1)
            ops[0]._is_uploaded = True

        def create_commit(self, operations, **kwargs):
            commits.append([op.path_in_repo for op in operations])
            return SimpleNamespace(oid="commit")

    monkeypatch.setattr(
        parallel,
        "wait",
        lambda active, timeout, return_when: real_wait(
            active, timeout=0.01, return_when=return_when
        ),
    )
    parallel.upload_parallel(
        API(),
        {"repo_id": "owner/data", "repo_type": "dataset", "title": "Data"},
        tmp_path,
        files,
        hashes,
        {"uploaded": {}},
        tmp_path / "upload_state.json",
        workers=2,
        batch_size=2,
    )
    assert commits == [["first.bin", "slow.bin"]]
