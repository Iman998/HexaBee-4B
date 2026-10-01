"""Manifest uploads with resumable batches and remote file verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from ..io import atomic_json, file_hash


class RetryableUploadError(RuntimeError):
    """A transfer exhausted its local retries and may be safely resumed."""


def authenticate(owner: str, token_file: Path | None = None):
    from huggingface_hub import HfApi, get_token

    token = (
        token_file.read_text().strip() if token_file else os.environ.get("HF_TOKEN") or get_token()
    )
    if not token:
        raise ValueError("Authenticate with hf auth login or set HF_TOKEN")
    api = HfApi(token=token)
    identity = api.whoami()
    if identity["name"] != owner:
        raise ValueError(f"Expected authenticated owner {owner}; refusing a different account")
    return api


def validate_release(folder: Path, expected_id: str) -> dict:
    from huggingface_hub import DatasetCard, ModelCard

    manifest = json.loads((folder / "release_manifest.json").read_text())
    if not manifest.get("complete") or manifest["repo_id"] != expected_id:
        raise ValueError("Incomplete or mismatched release manifest")
    card = DatasetCard if manifest["repo_type"] == "dataset" else ModelCard
    parsed = card.load(folder / "README.md")
    if not parsed.data.language:
        raise ValueError("Card must declare language metadata")
    for name, entry in manifest["files"].items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any(p.startswith(".") for p in relative.parts)
        ):
            raise ValueError("Unsafe release path")
        path = folder / relative
        if (
            not path.resolve().is_relative_to(folder.resolve())
            or not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or file_hash(path) != entry["sha256"]
        ):
            raise ValueError(f"Release file differs from manifest: {name}")
        if (
            path.suffix in {".md", ".json", ".yaml", ".py", ".txt"}
            and path.stat().st_size < 20_000_000
        ):
            text = path.read_text()
            if re.search(r"hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{20,}", text):
                raise ValueError(f"Possible embedded credential in {name}")
    return manifest


def remote_matches(local: Path, remote_file, sha256: str) -> bool:
    """Match the actual remote object, including when a saved upload state is stale."""
    if remote_file is None or remote_file.size != local.stat().st_size:
        return False
    if remote_file.lfs:
        return remote_file.lfs.sha256 == sha256
    digest = hashlib.sha1(b"blob " + str(local.stat().st_size).encode() + b"\0")
    with local.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return remote_file.blob_id == digest.hexdigest()


def is_transient_upload_error(exc: Exception) -> bool:
    """Recognize transport failures even when the Hub wraps them in RuntimeError."""
    import httpx

    if isinstance(exc, RetryableUploadError):
        return True
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, httpx.TransportError):
            return True
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
        if code in {408, 429, 500, 502, 503, 504}:
            return True
        if code == 400 and "<Code>RequestTimeout</Code>" in response.text:
            return True
        if code in {400, 401, 403, 404, 422}:
            return False
        exc = exc.__cause__ or exc.__context__
    return False


def publish(config: dict, key: str, execute: bool = False, token_file: Path | None = None) -> dict:
    import httpx
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    entry = next(e for e in config["repositories"] if e["key"] == key)
    folder = Path(config["staging_root"]) / key
    manifest = validate_release(folder, entry["repo_id"])
    names = list(manifest["files"]) + ["release_manifest.json"]
    if not execute:
        return {
            "dry_run": True,
            "repo_id": entry["repo_id"],
            "files": len(names),
            "bytes": sum(v["bytes"] for v in manifest["files"].values()),
        }
    if "upload_policy" in config:
        from .transport import configure_transport

        configure_transport(config["upload_policy"])
    api = authenticate(config["owner"], token_file)
    state_path = Path(config["scratch_root"]) / key / "upload_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"uploaded": {}}
    try:
        api.repo_info(entry["repo_id"], repo_type=entry["repo_type"])
        created = False
    except RepositoryNotFoundError:
        api.create_repo(entry["repo_id"], repo_type=entry["repo_type"], private=True)
        created = True
    # Uploaded batches remain private until complete on newly created repositories.
    expected_hash = {
        name: manifest["files"].get(name, {}).get("sha256") or file_hash(folder / name)
        for name in names
    }
    remote = {
        p.rfilename: p
        for p in api.repo_info(
            entry["repo_id"], repo_type=entry["repo_type"], files_metadata=True
        ).siblings
    }
    if state.get("repo_id", entry["repo_id"]) != entry["repo_id"]:
        raise ValueError("Upload state belongs to a different repository")
    pending = [
        name
        for name in names
        if not remote_matches(folder / name, remote.get(name), expected_hash[name])
    ]
    state["repo_id"] = entry["repo_id"]
    # Reconcile from server hashes so restarts report already committed files accurately.
    state["uploaded"] = {n: expected_hash[n] for n in names if n not in pending}
    atomic_json(state_path, state)
    revision = None
    batch_size = int(entry.get("upload_batch_files", 20))
    workers = int(entry.get("upload_workers", 4))
    if not 1 <= batch_size <= 100 or not 1 <= workers <= 16:
        raise ValueError("Upload batch must be 1–100 files and workers 1–16")
    if entry.get("upload_mode") == "parallel":
        from .parallel import upload_parallel

        upload_parallel(
            api, entry, folder, pending, expected_hash, state, state_path, workers, batch_size
        )
        pending = []
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        for attempt in range(3):
            try:
                revision = api.create_commit(
                    repo_id=entry["repo_id"],
                    repo_type=entry["repo_type"],
                    operations=[
                        CommitOperationAdd(path_in_repo=name, path_or_fileobj=folder / name)
                        for name in batch
                    ],
                    commit_message=f"Release {entry['title']} ({start + len(batch)}/{len(pending)} files)",
                    num_threads=workers,
                ).oid
                break
            except (HfHubHTTPError, RuntimeError, httpx.TransportError) as exc:
                transient = is_transient_upload_error(exc)
                if not transient or attempt == 2:
                    error_type = RetryableUploadError if transient else RuntimeError
                    raise error_type(
                        f"Upload failed for {entry['repo_id']} ({type(exc).__name__}); "
                        "completed batches are resumable. No repository visibility was changed."
                    ) from None
                print(
                    json.dumps(
                        {
                            "retry": entry["repo_id"],
                            "attempt": attempt + 1,
                            "reason": "temporary transfer failure",
                        }
                    ),
                    flush=True,
                )
                time.sleep(2**attempt)
        state["uploaded"].update({name: expected_hash[name] for name in batch})
        atomic_json(state_path, state)
        print(
            json.dumps(
                {"uploaded": entry["repo_id"], "batch_files": len(batch), "commit": revision}
            ),
            flush=True,
        )
    remote_info = api.repo_info(entry["repo_id"], repo_type=entry["repo_type"], files_metadata=True)
    remote = {p.rfilename: p for p in remote_info.siblings}
    for name in names:
        if not remote_matches(folder / name, remote.get(name), expected_hash[name]):
            raise RuntimeError(f"Remote file verification failed: {name}")
    if config.get("public", True):
        api.update_repo_settings(entry["repo_id"], repo_type=entry["repo_type"], private=False)
    result = {
        "published": True,
        "repo_id": entry["repo_id"],
        "repo_type": entry["repo_type"],
        "revision": remote_info.sha,
        "files": len(names),
        "new_repository": created,
        "public": config.get("public", True),
    }
    atomic_json(Path(config["scratch_root"]) / key / "published.json", result)
    return result
