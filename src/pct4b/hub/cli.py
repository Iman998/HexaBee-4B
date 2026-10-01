"""Prepare, validate, and publish independent HexaBee Hugging Face repositories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["list", "prepare", "refresh-card", "validate", "upload", "collection"]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Supervise uploads and restart after failures until complete or stopped",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        help="Optional private credential file outside the release folder",
    )
    args = parser.parse_args()
    if args.persistent and (args.action != "upload" or not args.execute):
        parser.error("--persistent requires upload --execute")
    config = json.loads(args.config.read_text())
    if args.action == "collection":
        from .collection import publish_collection

        if args.only:
            parser.error("The collection includes the complete configured release; omit --only")
        print(json.dumps(publish_collection(config, args.execute, args.token_file)))
        return
    entries = config["repositories"]
    if args.only:
        missing = set(args.only) - {e["key"] for e in entries}
        if missing:
            parser.error(f"Unknown repository keys: {sorted(missing)}")
        entries = [e for e in entries if e["key"] in args.only]
    for entry in entries:
        if args.action == "list":
            print(json.dumps({k: entry[k] for k in ("key", "repo_id", "repo_type", "kind")}))
        elif args.action == "prepare":
            from .staging import prepare

            prepare(config, entry["key"])
        elif args.action == "refresh-card":
            from .staging import refresh_card

            refresh_card(config, entry["key"])
            print(json.dumps({"card_refreshed": entry["repo_id"]}))
        elif args.action == "validate":
            from .publish import validate_release

            validate_release(Path(config["staging_root"]) / entry["key"], entry["repo_id"])
            print(json.dumps({"validated": entry["repo_id"]}))
        elif args.persistent:
            from .supervisor import supervise

            print(json.dumps(supervise(args.config, entry["key"], args.token_file)), flush=True)
        else:
            from .publish import publish

            try:
                print(
                    json.dumps(publish(config, entry["key"], args.execute, args.token_file)),
                    flush=True,
                )
            except Exception as exc:
                from ..io import atomic_json
                from .publish import is_transient_upload_error
                from .transport import redact

                temporary = is_transient_upload_error(exc)
                error = {
                    "state": "retryable_error" if temporary else "blocked",
                    "repo_id": entry["repo_id"],
                    "error_type": type(exc).__name__,
                    "message": redact(str(exc))[:600],
                }
                if args.execute:
                    atomic_json(
                        Path(config["scratch_root"]) / entry["key"] / "last_error.json", error
                    )
                print(json.dumps(error), flush=True)
                raise SystemExit(75 if temporary else 78) from None


if __name__ == "__main__":
    main()
