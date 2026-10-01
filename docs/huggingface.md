# Publishing the HexaBee release

The release has **11 independent repositories**: two merged models, three LoRA adapters, three data stages, one prompt inventory, one prediction archive, and one evaluation archive. Every repository has its own card, provenance, measured counts, license notice, and SHA-256 manifest.

## Install and configure

```bash
python -m pip install -e '.[hub,metrics,plots]'
mkdir -p runs
cp configs/hub/release.json runs/release.local.json
```

Edit the copy to point `source` and `sources[].path` at your archived artifacts. Relative paths resolve from the working directory. The committed configuration uses portable example paths; it contains no local credentials. The bundled artwork is selected by `brand_assets` and each repository's `banner`; matching overview panels are packaged with the Python source. To override either image, put `banner.png` or `overview.png` at `<visual_assets>/<repository-key>/` and set `visual_assets` in the local config.

```bash
hexabee-hub list --config runs/release.local.json
hexabee-hub prepare --config runs/release.local.json --only prompts
hexabee-hub validate --config runs/release.local.json --only prompts
hexabee-hub upload --config runs/release.local.json --only prompts
```

The final command is a **dry run**. It verifies the local manifest and reports the exact file inventory and byte count; it does not authenticate, create a repository, or upload.

The source-tree equivalent is `python scripts/hub_release.py ...`. `hexabee` is the primary research CLI; the historical `pct4b` command and Python namespace remain compatible with the archived experiments.

## Authenticate and publish

Use a token with repository write access for the configured owner. Authenticate outside the repository:

```bash
hf auth login
hexabee-hub upload --config runs/release.local.json --only prompts --execute
```

`HF_TOKEN` is also supported. For an automated job, `--token-file /private/path/token` reads a credential file outside the staging directory. Never add a token to a JSON configuration or commit it to Git. The uploader checks the authenticated username against `owner`.

The committed release configuration keeps every repository **private** (`"public": false`). New repositories are created privately, and an existing repository is made private before any upload begins. The uploader sends only files listed in the manifest, in batches of 20, then verifies remote size and LFS SHA-256 or Git object hash. Re-running the same command compares actual remote objects and resumes missing or changed files. Local batch state is diagnostic, not sufficient proof that a remote file still matches. Change `public` only when you intentionally approve a public release.

Preparation resumes at **completed source partitions**, validating their shard hashes. An interrupted partial partition is rebuilt. Changing a completed source's size/mtime, schema, or conversion recipe requires a new destination. Preparation is streaming and produces Zstandard-compressed Parquet shards of up to 25,000 rows. Reserve disk for compressed data, model staging if hard links are unavailable, and temporary partial partitions.

## Release map

| Key | Repository | Unit / exact base |
|:--|:--|:--|
| `model` | [HexaBee-4B](https://huggingface.co/Iman998/HexaBee-4B) | Final merged translator |
| `multilingual-base` | [Multilingual Base](https://huggingface.co/Iman998/HexaBee-4B-Multilingual-Base) | Merged PT → multilingual SFT |
| `lora-pt` | [PT LoRA](https://huggingface.co/Iman998/HexaBee-4B-PT-LoRA) | `google/gemma-3-4b-it` |
| `lora-sft` | [Multilingual SFT LoRA](https://huggingface.co/Iman998/HexaBee-4B-Multilingual-SFT-LoRA) | `google/gemma-3-4b-it`; PT already continued |
| `lora-translation` | [Translation LoRA](https://huggingface.co/Iman998/HexaBee-4B-Translation-LoRA) | `Iman998/HexaBee-4B-Multilingual-Base` |
| `data-pt` | [Wikipedia PT](https://huggingface.co/datasets/Iman998/HexaBee-Wikipedia-PT) | Raw text rows; five registered languages |
| `data-sft` | [Multilingual SFT](https://huggingface.co/datasets/Iman998/HexaBee-Multilingual-SFT) | 30,153,404 records in 22 configurations |
| `data-translation` | [Bidirectional Translation](https://huggingface.co/datasets/Iman998/HexaBee-Bidirectional-Translation) | Aligned pairs + two overlapping SFT views |
| `prompts` | [System Prompts](https://huggingface.co/datasets/Iman998/HexaBee-System-Prompts) | 341,142 original pool rows |
| `outputs` | [Model Outputs](https://huggingface.co/datasets/Iman998/HexaBee-Model-Outputs) | Three full-test + three judged configurations |
| `evaluation` | [Evaluation](https://huggingface.co/datasets/Iman998/HexaBee-Evaluation) | Metrics, uncertainty, and composition audit |

Do not stack the PT and multilingual SFT adapters: the latter is a continued adapter. The final translation adapter uses the merged multilingual base, not the original Google checkpoint.

## Publish each group

```bash
hexabee-hub prepare --config runs/release.local.json --only model multilingual-base lora-pt lora-sft lora-translation
hexabee-hub prepare --config runs/release.local.json --only data-pt data-sft data-translation
hexabee-hub prepare --config runs/release.local.json --only prompts outputs evaluation
hexabee-hub validate --config runs/release.local.json
hexabee-hub upload --config runs/release.local.json --execute
```

Only run one uploader per repository at a time. Do not modify a staged release while its upload is running. Failed authentication or permission errors stop that repository; completed batches remain available for the next run. Publication does not delete unrelated remote files.

## What is preserved

- Original strings, empty SFT outputs, train/test membership, and prompt multiplicities.
- Actual saved LoRA configuration and Safetensors bytes; only local path metadata is normalized.
- Full-test predictions separately from the matched 300-item judged subset.
- Raw judge response objects, including recoverable zero scores.
- Dataset licenses and upstream provenance as recorded; no blanket software license is assigned to all text or weights.

Per-row language tags removed by the original phase-2 export remain null. IDs for PT/SFT rows incorporate source component, position, and content. They distinguish exported rows and are not Wikipedia article identifiers. Source partition fingerprints use size/mtime and recipe metadata; published files have full SHA-256 checksums.

The PT corpus contains five registered language sources; the final translation evaluation covers six languages. The four historical Persian registrations point at one source, released once. Raw PT rows differ from the trainer's packed example count. The translation pair view overlaps the forward and reverse SFT views, so their sizes must not be added as independent training examples.

## Reproducibility and verification

`release_stats.json` and `release_manifest.json` are distributed with each repository. Local `published.json` records the verified Hub revision, file count, and visibility. The code ZIP excludes local runs, credentials, weight files, raw corpora, and API caches. Tests use mock Hub operations and temporary miniature Parquet data; they never publish anything.


## Update cards and create a collection

After an upload has finished, edit the card descriptions in your configuration and run:

```bash
hexabee-hub refresh-card --config runs/release.local.json --only model
hexabee-hub upload --config runs/release.local.json --only model --execute
```

Card refresh preserves existing data and weight shards and updates the manifest for changed metadata. The collection command requires every configured repository to be verified and public, so defer it until the public release is approved. Card upload remains a dry run unless `--execute` is present.

The Hub card builder stages the included banner and overview images during preparation or refresh. `scripts/build_hub_card_visuals.py` can regenerate the overview panels with Pillow. Evaluation figures are included in `figures/` and can be regenerated with `python scripts/build_paper_figures.py` inside a Slurm CPU job. The images support navigation and explanation; the numeric evidence remains in `results/paper/`.


If Xet uploads repeatedly time out on a network, stop that uploader before retrying the same repository through the supported HTTP/LFS path:

```bash
HF_HUB_DISABLE_XET=1 hexabee-hub upload --config runs/release.local.json --only data-pt --execute
```

Successful remote files are verified and reused. This changes the transfer protocol, not the released artifact bytes.

## Persistent uploads on unreliable connections

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  hexabee-hub upload --config runs/release.local.json --only data-sft --execute --persistent
```

`--persistent` starts one supervised worker per selected repository, processing repository selections in order. It uses an exclusive lock to prevent duplicate supervisors. The worker retains verified remote objects and commits small batches; failed attempts do not reset completed uploads. Each worker gets three immediate attempts for temporary batch errors. The supervisor restarts a failed worker indefinitely, with a 30-second delay increasing to at most five minutes. An access or integrity error is recorded as `waiting_for_fix` and checked again every 15 minutes, so an operator can correct the token or local configuration without restarting the supervisor.

The `upload_policy` configuration controls these defaults:

| Setting | Default | Meaning |
|:--|--:|:--|
| `connect_timeout_seconds` | 180 | Connection and connection-pool timeout |
| `transfer_timeout_seconds` | 1800 | Read/write inactivity timeout, including calls with short Hub defaults |
| `watchdog_seconds` | 3600 | Restart a worker after this interval without a newly committed batch |
| `restart_delay_seconds` | 30 | Initial delay before restarting a failed worker |
| `max_restart_delay_seconds` | 300 | Maximum delay for ordinary restarts |
| `blocked_retry_seconds` | 900 | Recheck interval for errors requiring a fix |

Each repository may set `upload_batch_files` (default 20) and `upload_workers` (default 4). The phase-2 SFT release uses 40 files per commit and four parallel LFS transfers. The uploader waits for preuploads to fill a batch, reducing commit-rate throttling; a 429 response may still require waiting for the Hub quota window to reset. The HTTP/LFS path supports the explicit timeouts; Xet has its own networking stack. A remote storage service can still close an idle socket independently of client settings; the supervisor resumes after that failure.

The work directory contains:

- `supervisor_status.json`: heartbeat every 30 seconds, worker PID, retry state, and committed byte/file counts.
- `attempt-*.log`: individual worker logs; HTTP diagnostics redact signed URLs and credentials.
- `last_error.json`: a concise failure classification.
- `upload_state.json`: hashes of files in successful commits.
- `published.json`: final remotely verified revision and visibility.

Progress measures committed bytes, not bytes temporarily sent to an uncommitted upload. A large batch or stalled network can therefore show zero committed progress during transfer.

For an unattended shell session, use `nohup` and keep the credential outside the project:

```bash
nohup env HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  hexabee-hub upload --config runs/release.local.json --only data-sft --execute --persistent \
  > runs/upload-supervisor.log 2>&1 < /dev/null &
echo $! > runs/upload-supervisor.pid
```

SIGTERM or Ctrl-C intentionally stops the supervisor and its worker process group. Re-run the same command to resume. Detached execution survives terminal closure; machine/container shutdown still requires relaunching the command. Repository visibility follows `public` in the configuration; the supervisor does not change this policy.
