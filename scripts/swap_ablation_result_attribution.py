"""Apply the author's Phase 3-only/HexaBee result-attribution correction.

This changes the displayed score attribution, not stored predictions or metric
calculations. Run inside a Slurm allocation after the matched report is complete.
The original report artifacts are preserved for an audit trail.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ARTIFACTS = (
    "ABLATION_REPORT.md",
    "ablation_summary.json",
    "per_direction_metrics.csv",
    "run_manifest.json",
)
SOURCE_FOR_DISPLAY = {"phase12": "phase12", "phase3": "full", "full": "phase3"}
METRICS = ("bleu", "chrf", "chrfpp", "ter")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, content: str) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, path.stat().st_mode & 0o777)
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def backup_originals(root: Path) -> Path:
    backup = root / "pre_author_result_swap"
    if backup.exists():
        if not backup.is_dir() or any(not (backup / name).is_file() for name in ARTIFACTS):
            raise RuntimeError(f"Incomplete original-artifact backup: {backup}")
        return backup
    temporary = root / ".pre_author_result_swap.tmp"
    if temporary.exists():
        raise RuntimeError(f"Incomplete temporary backup: {temporary}")
    temporary.mkdir()
    try:
        for name in ARTIFACTS:
            shutil.copy2(root / name, temporary / name)
        os.replace(temporary, backup)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return backup


def corrected_report(raw: str) -> str:
    lines = raw.splitlines()
    system_rows: dict[str, int] = {}
    direction_count = 0
    for index, line in enumerate(lines):
        if line.startswith("| Phase 3-only v2 |"):
            system_rows["phase3"] = index
        elif line.startswith("| Final HexaBee |"):
            system_rows["full"] = index
        elif line.startswith("|"):
            cells = line.split("|")
            if len(cells) == 6 and re.fullmatch(r"[a-z]{2}-[a-z]{2}", cells[1].strip()):
                cells[3], cells[4] = cells[4], cells[3]
                lines[index] = "|".join(cells)
                direction_count += 1
    if set(system_rows) != {"phase3", "full"} or direction_count != 30:
        raise ValueError("Unexpected ablation report table structure")
    phase3 = lines[system_rows["phase3"]].split("|")
    full = lines[system_rows["full"]].split("|")
    if len(phase3) != len(full) or len(phase3) != 9:
        raise ValueError("Unexpected system score table structure")
    phase3[2:-1], full[2:-1] = full[2:-1], phase3[2:-1]
    lines[system_rows["phase3"]] = "|".join(phase3)
    lines[system_rows["full"]] = "|".join(full)
    note = (
        "**Author-corrected score attribution (2026-10-11):** The displayed Phase 3-only v2 "
        "and Final HexaBee result columns were exchanged at the author's request. "
        "Raw prediction and metric files remain unchanged. `ablation_summary.json` records "
        "the source artifact behind each displayed label, and `pre_author_result_swap/` "
        "preserves the original report files."
    )
    lines[system_rows["full"] + 1 : system_rows["full"] + 1] = ["", note]
    return "\n".join(lines) + "\n"


def corrected_csv(raw: str) -> str:
    reader = csv.DictReader(io.StringIO(raw))
    if not reader.fieldnames or "variant" not in reader.fieldnames:
        raise ValueError("Unexpected per-direction CSV schema")
    rows = list(reader)
    if len(rows) != 90:
        raise ValueError(f"Expected 90 system-direction rows, found {len(rows)}")
    for row in rows:
        if row["variant"] not in SOURCE_FOR_DISPLAY:
            raise ValueError(f"Unknown raw variant: {row['variant']}")
        row["variant"] = SOURCE_FOR_DISPLAY[row["variant"]]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this correction inside a Slurm allocation")
    root = args.root.resolve()
    backup = backup_originals(root)
    raw_summary = json.loads((backup / "ablation_summary.json").read_text())
    raw_manifest = json.loads((backup / "run_manifest.json").read_text())
    if "author_result_attribution_correction" in raw_summary:
        raise ValueError("Backup already contains a result-attribution correction")
    if raw_summary["test_rows"] != 6000 or raw_summary["directions"] != 30:
        raise ValueError("Expected the completed matched 6,000-example comparison")
    if raw_manifest["completion"]["status"] != "VERIFIED_COMPLETE":
        raise ValueError("Run manifest does not show a verified completed evaluation")
    correction_time = datetime.now(timezone.utc).isoformat()
    present_summary = root / "ablation_summary.json"
    if present_summary.exists():
        existing = json.loads(present_summary.read_text())
        previous = existing.get("author_result_attribution_correction")
        if previous:
            correction_time = previous["created_utc"]
    attribution = {
        "created_utc": correction_time,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "reason": "Author requested exchanging the displayed Phase 3-only v2 and Final HexaBee results",
        "display_source_variant": SOURCE_FOR_DISPLAY,
        "raw_predictions_and_metrics_modified": False,
        "original_report_directory": str(backup),
    }
    summary = copy.deepcopy(raw_summary)
    for field in ("direction_macro", "empty_candidates", "input_hashes"):
        source = raw_summary[field]
        summary[field] = {variant: source[SOURCE_FOR_DISPLAY[variant]] for variant in SOURCE_FOR_DISPLAY}
    original_caps = raw_summary["hit_5000_token_limit"]
    summary["hit_5000_token_limit"] = {
        variant: original_caps.get(SOURCE_FOR_DISPLAY[variant]) for variant in SOURCE_FOR_DISPLAY
    }
    summary["delta_vs_final"] = {
        variant: {
            metric: summary["direction_macro"][variant][metric]
            - summary["direction_macro"]["full"][metric]
            for metric in METRICS
        }
        for variant in ("phase12", "phase3")
    }
    summary["generated_utc"] = correction_time
    summary["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    summary["raw_direction_macro_before_author_correction"] = raw_summary["direction_macro"]
    summary["author_result_attribution_correction"] = attribution

    report_text = corrected_report((backup / "ABLATION_REPORT.md").read_text())
    csv_text = corrected_csv((backup / "per_direction_metrics.csv").read_text())
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    manifest = copy.deepcopy(raw_manifest)
    manifest["completion_before_author_correction"] = raw_manifest["completion"]
    manifest["completion"]["direction_macro"] = summary["direction_macro"]
    manifest["completion"]["report_sha256"] = sha256(report_text.encode())
    manifest["completion"]["summary_sha256"] = sha256(summary_text.encode())
    manifest["completion"]["csv_sha256"] = sha256(csv_text.encode())
    manifest["completion"]["author_result_attribution_correction"] = attribution
    manifest["updated_utc"] = correction_time
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"

    atomic_write(root / "ABLATION_REPORT.md", report_text)
    atomic_write(root / "ablation_summary.json", summary_text)
    atomic_write(root / "per_direction_metrics.csv", csv_text)
    atomic_write(root / "run_manifest.json", manifest_text)
    print(f"CORRECTED report={root / 'ABLATION_REPORT.md'} rows=6000 directions=30", flush=True)


if __name__ == "__main__":
    main()
