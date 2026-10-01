"""Create a reproducible source ZIP using an explicit, small release allowlist."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from itertools import chain
from pathlib import Path

TOP_LEVEL = {
    "README.md",
    "LICENSE",
    "DATA_AND_MODELS.md",
    "CITATION.cff",
    "pyproject.toml",
    ".gitignore",
    ".env.example",
    "references.bib",
    "requirements-tested.txt",
}
DIRECTORIES = {
    "src",
    "configs",
    "scripts",
    "docs",
    "tests",
    "examples",
    "results",
    "figures",
    "paper",
    "assets",
    ".github",
}
SUFFIXES = {
    ".py",
    ".json",
    ".jsonl",
    ".csv",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".pdf",
    ".svg",
    ".png",
    ".zip",
}


def release_files(root: Path) -> list[Path]:
    selected = []
    candidates = chain(
        (root / name for name in TOP_LEVEL),
        *(sorted((root / folder).rglob("*")) for folder in sorted(DIRECTORIES)),
    )
    for path in sorted(candidates):
        relative = path.relative_to(root)
        if path.is_symlink() or not path.is_file():
            continue
        if len(relative.parts) == 1 and path.name in TOP_LEVEL:
            selected.append(path)
        elif (
            relative.parts[0] in DIRECTORIES
            and path.is_file()
            and path.suffix in SUFFIXES
            and not any(p.startswith(".") or p == "__pycache__" for p in relative.parts[1:])
            and not any(p.endswith(".egg-info") for p in relative.parts)
        ):
            selected.append(path)
    return selected


def package(root: Path, output: Path) -> int:
    files = release_files(root)
    if not files:
        raise ValueError("No release files found")
    manifest = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in files:
                name = path.relative_to(root).as_posix()
                data = path.read_bytes()
                manifest[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                entry = zipfile.ZipInfo(
                    "hexabee-reproducibility/" + name, date_time=(2026, 10, 2, 0, 0, 0)
                )
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, data)
            entry = zipfile.ZipInfo(
                "hexabee-reproducibility/RELEASE_MANIFEST.json", date_time=(2026, 10, 2, 0, 0, 0)
            )
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    count = package(root, args.output)
    print(f"Packaged {count} files: {args.output}")


if __name__ == "__main__":
    main()
