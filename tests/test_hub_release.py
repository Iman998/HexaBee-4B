"""Publication regression tests; never contact the Hub."""

# ruff: noqa: E402 -- optional Hub dependencies are skipped before module imports.

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")
pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from pct4b.hub.cards import build_card
from pct4b.hub.parquet import prepare_shards
from pct4b.hub.publish import validate_release
from pct4b.hub.staging import sanitize_config, stage_model, visual_sources
from pct4b.io import atomic_json, write_records


def test_shards_preserve_empty_and_exact_strings(tmp_path):
    source = tmp_path / "input.json"
    rows = [
        {"system": "Answer in Persian.", "instruct": " Q  ", "output": ""},
        {"system": "s", "instruct": "q", "output": "a"},
    ]
    write_records(source, rows)
    target = tmp_path / "shards"
    result = prepare_shards(source, target, "sft", {"component": "qa_a", "task": "complete"}, 1, 2)
    assert result["rows"] == 2 and len(result["files"]) == 2
    loaded = pq.read_table(target / "part-00000.parquet").to_pylist()[0]
    assert loaded["output"] == "" and loaded["instruct"] == " Q  "
    assert loaded["source_language"] is None
    assert (
        prepare_shards(source, target, "sft", {"component": "qa_a", "task": "complete"}, 1, 2)
        == result
    )


def test_corrupt_completed_shard_is_not_silently_reused(tmp_path):
    source = tmp_path / "input.jsonl"
    write_records(source, [{"text": "hello"}])
    dest = tmp_path / "out"
    prepare_shards(source, dest, "text", {"component": "wiki_en", "language": "en"}, 1, 1)
    (dest / "part-00000.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupted"):
        prepare_shards(source, dest, "text", {"component": "wiki_en", "language": "en"}, 1, 1)


def test_wrong_count_never_marks_partition_complete(tmp_path):
    source = tmp_path / "input.jsonl"
    write_records(source, [{"text": "hello"}])
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="Count mismatch"):
        prepare_shards(source, dest, "text", {"component": "wiki_en", "language": "en"}, 1, 2)
    assert not (dest / "partition.json").exists()


def test_changed_source_requires_new_destination(tmp_path):
    source = tmp_path / "input.jsonl"
    write_records(source, [{"text": "hello"}])
    dest = tmp_path / "out"
    prepare_shards(source, dest, "text", {"component": "wiki_en", "language": "en"})
    write_records(source, [{"text": "a changed source"}])
    with pytest.raises(ValueError, match="Changed input"):
        prepare_shards(source, dest, "text", {"component": "wiki_en", "language": "en"})


def test_adapter_public_base_and_no_unsafe_trainer_artifacts(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    dest.mkdir()
    atomic_json(
        source / "adapter_config.json", {"base_model_name_or_path": "/private/base", "r": 256}
    )
    (source / "adapter_model.safetensors").write_bytes(b"fixture-only-weight-file")
    (source / "training_args.bin").write_bytes(b"do-not-release")
    stage_model(
        {
            "source": str(source),
            "kind": "adapter",
            "base_model": "Iman998/HexaBee-4B-Multilingual-Base",
        },
        dest,
    )
    assert (
        json.loads((dest / "adapter_config.json").read_text())["base_model_name_or_path"]
        == "Iman998/HexaBee-4B-Multilingual-Base"
    )
    assert not (dest / "training_args.bin").exists()


def test_model_index_missing_shard_fails(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    dest.mkdir()
    (source / "model-00001-of-00002.safetensors").write_bytes(b"fixture")
    atomic_json(
        source / "model.safetensors.index.json",
        {"weight_map": {"a": "model-00002-of-00002.safetensors"}},
    )
    with pytest.raises(ValueError, match="missing shard"):
        stage_model(
            {"source": str(source), "kind": "model", "base_model": "google/gemma-3-4b-it"}, dest
        )


def test_dataset_card_has_standalone_configuration_and_not_merged_scope():
    entry = {
        "title": "HexaBee Prompts",
        "repo_id": "Iman998/HexaBee-System-Prompts",
        "repo_type": "dataset",
        "kind": "prompts",
        "scope": "prompt rows",
        "subtitle": "Instructions",
        "description": "Original pools.",
        "contents": "Prompts.",
        "provenance": "Archived pools.",
        "limitations": "Duplicate wording retained.",
    }
    text = build_card(
        entry,
        [entry],
        {"rows": 341142},
        [
            {
                "config_name": "prompts",
                "data_files": [{"split": "train", "path": "data/prompts/train/*.parquet"}],
            }
        ],
    )
    assert "341,142" in text and "config_name: prompts" in text
    assert 'load_dataset("Iman998/HexaBee-System-Prompts", "prompts"' in text
    assert "TODO" not in text
    assert "assets/banner.png" not in text
    assert "assets/overview.png" not in text


def test_release_config_has_bundled_visuals_for_every_card():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/hub/release.json").read_text())
    for entry in config["repositories"]:
        assert set(visual_sources(config, entry)) == {"banner.png", "overview.png"}


def test_release_path_traversal_is_rejected(tmp_path):
    (tmp_path / "README.md").write_text("---\nlanguage: [en]\n---\n# Test\n")
    atomic_json(
        tmp_path / "release_manifest.json",
        {
            "complete": True,
            "repo_id": "a/b",
            "repo_type": "dataset",
            "files": {"../secret": {"bytes": 1, "sha256": "bad"}},
        },
    )
    with pytest.raises(ValueError, match="Unsafe"):
        validate_release(tmp_path, "a/b")


def test_manifest_detects_mutated_card(tmp_path):
    card = tmp_path / "README.md"
    card.write_text("---\nlanguage: [en]\n---\n# Test\n")
    data = card.read_bytes()
    atomic_json(
        tmp_path / "release_manifest.json",
        {
            "complete": True,
            "repo_id": "a/b",
            "repo_type": "dataset",
            "files": {
                "README.md": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            },
        },
    )
    validate_release(tmp_path, "a/b")
    card.write_text(card.read_text() + "changed")
    with pytest.raises(ValueError, match="differs"):
        validate_release(tmp_path, "a/b")


def test_metadata_removes_local_model_path():
    assert sanitize_config({"_name_or_path": "/data/a", "nested": {"source": "/mnt/a/b"}}) == {
        "nested": {"source": "b"}
    }


def test_remote_verification_detects_same_size_changed_content(tmp_path):
    from types import SimpleNamespace

    from pct4b.hub.publish import remote_matches

    path = tmp_path / "file"
    path.write_bytes(b"abcd")
    sha = hashlib.sha256(b"abcd").hexdigest()
    old = SimpleNamespace(size=4, lfs=SimpleNamespace(sha256=hashlib.sha256(b"dcba").hexdigest()))
    assert not remote_matches(path, old, sha)
    current = SimpleNamespace(size=4, lfs=None, blob_id=hashlib.sha1(b"blob 4\0abcd").hexdigest())
    assert remote_matches(path, current, sha)
    assert not remote_matches(path, None, sha)


def test_symlink_outside_release_is_rejected(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("external")
    folder = tmp_path / "release"
    folder.mkdir()
    (folder / "README.md").write_text("---\nlanguage: [en]\n---\n# Test\n")
    (folder / "linked.txt").symlink_to(outside)
    atomic_json(
        folder / "release_manifest.json",
        {
            "complete": True,
            "repo_id": "a/b",
            "repo_type": "dataset",
            "files": {
                "linked.txt": {"bytes": 8, "sha256": hashlib.sha256(b"external").hexdigest()}
            },
        },
    )
    with pytest.raises(ValueError, match="differs"):
        validate_release(folder, "a/b")


def _fake_release(tmp_path):
    folder = tmp_path / "staging" / "test"
    folder.mkdir(parents=True)
    data = b"---\nlanguage: [en]\n---\n# Test\n"
    (folder / "README.md").write_bytes(data)
    atomic_json(
        folder / "release_manifest.json",
        {
            "complete": True,
            "repo_id": "a/b",
            "repo_type": "dataset",
            "files": {
                "README.md": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            },
        },
    )
    return {
        "owner": "a",
        "staging_root": str(tmp_path / "staging"),
        "scratch_root": str(tmp_path / "work"),
        "repositories": [
            {"key": "test", "repo_id": "a/b", "repo_type": "dataset", "title": "Test"}
        ],
    }, folder


def test_dry_run_never_authenticates(tmp_path, monkeypatch):
    from pct4b.hub import publish as publisher

    config, _ = _fake_release(tmp_path)

    def unexpected(*args):
        pytest.fail("Dry run attempted authentication")

    monkeypatch.setattr(publisher, "authenticate", unexpected)
    assert publisher.publish(config, "test")["dry_run"]


@pytest.mark.parametrize("corrupt", [False, True])
def test_publication_requires_verified_remote_bytes(tmp_path, monkeypatch, corrupt):
    from types import SimpleNamespace

    from pct4b.hub import publish as publisher

    config, _ = _fake_release(tmp_path)
    objects, visibility = {}, []

    class FakeApi:
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(siblings=list(objects.values()), sha="fixture-revision")

        def create_commit(self, **kwargs):
            for op in kwargs["operations"]:
                from pathlib import Path

                data = Path(op.path_or_fileobj).read_bytes()
                objects[op.path_in_repo] = SimpleNamespace(
                    rfilename=op.path_in_repo,
                    size=len(data),
                    lfs=None,
                    blob_id="corrupted"
                    if corrupt
                    else hashlib.sha1(
                        b"blob " + str(len(data)).encode() + b"\0" + data
                    ).hexdigest(),
                )
            return SimpleNamespace(oid="fixture-revision")

        def update_repo_settings(self, *args, **kwargs):
            visibility.append(kwargs["private"])

    monkeypatch.setattr(publisher, "authenticate", lambda *args: FakeApi())
    if corrupt:
        with pytest.raises(RuntimeError, match="verification failed"):
            publisher.publish(config, "test", execute=True)
        assert visibility == []
    else:
        result = publisher.publish(config, "test", execute=True)
        assert result["published"] and result["files"] == 2
        assert visibility == [False]


def test_ranked_hub_table_honors_ter_and_rounded_ties():
    from pct4b.hub.evidence import ranked_table

    rows = [{"model": "a", "ter": 2.0}, {"model": "b", "ter": 2.004}, {"model": "c", "ter": 3.0}]
    table = ranked_table(rows, ("ter",), ("TER ↓",))
    assert table.count("**2.00**") == 2
    assert "<u>3.00</u>" in table


def test_wrapped_s3_timeout_retries_but_access_denial_does_not():
    import httpx
    from huggingface_hub.errors import BadRequestError, HfHubHTTPError

    from pct4b.hub.publish import is_transient_upload_error

    request = httpx.Request("PUT", "https://example.org/object")
    timeout = BadRequestError(
        "timeout",
        response=httpx.Response(
            400, request=request, text="<Error><Code>RequestTimeout</Code></Error>"
        ),
    )
    wrapped = RuntimeError("upload failed")
    wrapped.__cause__ = timeout
    assert is_transient_upload_error(wrapped)
    denied = HfHubHTTPError("forbidden", response=httpx.Response(403, request=request))
    wrapped.__cause__ = denied
    assert not is_transient_upload_error(wrapped)
    assert not is_transient_upload_error(RuntimeError("unknown programming error"))


def test_upload_timeout_policy_overrides_short_per_request_defaults(monkeypatch):
    import httpx
    import huggingface_hub

    from pct4b.hub.transport import configure_transport, redact

    factories = []
    monkeypatch.setattr(huggingface_hub, "set_client_factory", factories.append)
    configure_transport({"connect_timeout_seconds": 180, "transfer_timeout_seconds": 1800})
    with factories[0]() as client:
        request = httpx.Request(
            "PUT", "https://example.org/file", extensions={"timeout": {"read": 10}}
        )
        for hook in client.event_hooks["request"]:
            hook(request)
        assert request.extensions["timeout"] == {
            "connect": 180,
            "read": 1800,
            "write": 1800,
            "pool": 180,
        }
    assert "secret" not in redact("Failed https://example.org/file?secret=abc")


@pytest.mark.parametrize("first_exit,expected_state", [(75, "retry_wait"), (78, "waiting_for_fix")])
def test_supervisor_restarts_failed_worker_and_keeps_visibility(
    tmp_path, monkeypatch, first_exit, expected_state
):
    from pct4b.hub import supervisor

    config, folder = _fake_release(tmp_path)
    config["public"] = False
    config["upload_policy"] = {
        "restart_delay_seconds": 0.01,
        "max_restart_delay_seconds": 0.01,
        "blocked_retry_seconds": 0.01,
    }
    config_path = tmp_path / "config.json"
    atomic_json(config_path, config)
    calls, events = [], []
    codes = iter([first_exit, 0])
    original_write = supervisor.atomic_json

    class Worker:
        pid = 999999

        def __init__(self, code):
            self.code = code

        def poll(self):
            return self.code

        def wait(self, timeout=None):
            return self.code

    def launch(command, **kwargs):
        calls.append(command)
        assert kwargs["start_new_session"]
        return Worker(next(codes))

    def record(path, value):
        events.append(value)
        original_write(path, value)

    monkeypatch.setattr(supervisor.subprocess, "Popen", launch)
    monkeypatch.setattr(supervisor, "atomic_json", record)
    result = supervisor.supervise(config_path, "test")
    assert result["state"] == "complete" and len(calls) == 2
    assert events[0]["state"] == expected_state
    assert json.loads(config_path.read_text())["public"] is False
    assert "--persistent" not in calls[0]


def test_supervisor_rejects_a_second_writer(tmp_path):
    import fcntl
    from pathlib import Path

    from pct4b.hub.supervisor import supervise

    config, _ = _fake_release(tmp_path)
    config_path = tmp_path / "config.json"
    atomic_json(config_path, config)
    scratch = Path(config["scratch_root"]) / "test"
    scratch.mkdir(parents=True)
    with (scratch / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another upload supervisor"):
            supervise(config_path, "test")


def test_watchdog_restarts_a_worker_without_commit_progress(tmp_path, monkeypatch):
    import itertools
    from types import SimpleNamespace

    from pct4b.hub import supervisor

    config, _ = _fake_release(tmp_path)
    config["upload_policy"] = {"watchdog_seconds": 1, "restart_delay_seconds": 0.01}
    path = tmp_path / "config.json"
    atomic_json(path, config)
    workers, events = [], []
    ticks = itertools.count()
    clock = SimpleNamespace(
        monotonic=lambda: next(ticks), sleep=lambda _: None, time_ns=lambda: next(ticks)
    )
    original_write = supervisor.atomic_json

    class Worker:
        pid = 999999

        def __init__(self, code):
            self.code = code

        def poll(self):
            return self.code

        def wait(self, timeout=None):
            return self.code

    def launch(*args, **kwargs):
        worker = Worker(None if not workers else 0)
        workers.append(worker)
        return worker

    def record(path, value):
        events.append(value)
        original_write(path, value)

    monkeypatch.setattr(supervisor, "time", clock)
    monkeypatch.setattr(supervisor.subprocess, "Popen", launch)
    monkeypatch.setattr(supervisor, "terminate_worker", lambda w: setattr(w, "code", -15))
    monkeypatch.setattr(supervisor, "atomic_json", record)
    assert supervisor.supervise(path, "test")["state"] == "complete"
    assert len(workers) == 2
    assert any(e.get("watchdog_restart") for e in events)
