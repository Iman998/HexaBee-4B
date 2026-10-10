# Checkpoint translation comparison

The current comparison uses a fixed **6,000-example subset** of the archived
36,572-example synthetic translation test: 200 examples from each of the 30
language directions. `scripts/build_ablation_subset.py` selects the 200 IDs
with the smallest SHA-256 values of `hexabee-ablation-20261010:<id>` in each
direction. It preserves the original source and reference text, and extracts
the published final-system outputs for those exact IDs. The selection manifest
records source-file hashes, the selected-ID hash, and all 30 counts.

All new predictions use the `archived-short` translation prompt, greedy
Transformers generation, eager attention, batch size 4, and a 5,000-token
ceiling. They are written to
`/mnt/ai_team/iman_barati/iman_31/paper/translate/ablation_eval/balanced6000/`.
The old full-test SQLite checkpoints are preserved under `ablation_eval/` and
compatible completed predictions are copied into the selected-run databases.
`scripts/ablation_infer.py` verifies checkpoint and decoding settings before
reusing any candidate.

| Variant | Weights | Role in the comparison |
|:--|:--|:--|
| Phases 1+2 | Released multilingual merged base | Intermediate checkpoint before translation specialization |
| Phase 3 only | Gemma 3 4B IT with `gemma-3-4b-translate-v2` LoRA | Independent translation-only comparator |
| Phases 1+2+3 | Archived published HexaBee outputs | Complete system |

## Phase 1 translation check

Before the 6,000-example run, all **360** Phase 1 predictions saved by the
stopped full-test job were compared with the same test identities from the
other checkpoints. These 360 examples cover all 30 directions, with 5–20
examples per direction. The direction-macro chrF++ scores were **4.87** for
Phase 1, **55.86** for phases 1+2, and **75.26** for phase 3 only. Phase 1
reached the 5,000-token ceiling on **141/360** examples; 25/30 direction
scores were below 10. Manual inspection found source copying, wrong target
languages, and unrelated generated articles. The script and saved audit are
`scripts/audit_phase1_outputs.py` and
`ablation_eval/balanced6000/phase1_triage.json` on the cluster. Raw Phase 1
outputs remain in `ablation_eval/phase1/progress.sqlite`.

Phase 1 is Wikipedia continued pretraining, before translation instruction
tuning. Its diagnostic outputs do not support treating it as a working
translator, so the user's requested 6,000-example automatic-metric comparison
excludes it. The 360 examples were saved in original test order before the
job stopped; this diagnostic is **not** a representative full-test estimate.
It still provides direct evidence that translation behavior emerges after the
later training stages.

## Interpretation limits

The phase-3-only v2 adapter uses rank 512 and alpha 1024; the final stage-3
adapter uses a different training configuration and is based on the phase-1+2
checkpoint. V2 is therefore a translation-only comparator, **not a controlled
removal** of phases 1 and 2 from the published final model. The archived
final outputs used a different serving engine from the new Transformers runs.
The final comparison must state both limitations.

These automatic metrics are BLEU, chrF, chrF++, and TER from the same `pct4b`
evaluator used for the release. Scores are computed per direction and averaged
equally over all 30 directions. Each variant must contain exactly the same
6,000 test identities before any comparison. LLM judging is outside this
ablation. The paper's original judged subset contained **300 matched test
examples per model**, ten per direction, or 900 output rows for three models.

## Reproduce on the Zharfatech cluster

Run every storage and compute operation through Slurm. The release data,
checkpoints, and scripts must be visible on the compute node. The CPU builder
job is `jobs/ablation_subset_6000.sbatch`; the H100 inference job is
`jobs/ablation_infer_6000.sbatch`. Use
`jobs/ablation_parallel_metrics.sbatch` for CPU scoring. It runs eight
independent direction scorers, saves each direction's sufficient statistics,
and resumes without recomputing finished directions. Its results matched the
published `pct4b.evaluation.score_metrics` JSON exactly in a Slurm smoke test
with two examples in each of the 30 directions. The earlier serial wrapper
`jobs/ablation_metrics_6000.sbatch` remains for provenance; it exceeded a
one-hour limit on the archived final outputs.

Job names are `eval+phase12`, `eval+phase3`, and `metric+...`. The checkpoint
metric jobs depend on successful inference. Each inference job checkpoints to
its own SQLite database after every batch; `predictions.jsonl` appears
atomically only when all 6,000 candidates exist.

The original Gemma checkpoint is `/mnt/ai_team/models/gemma-3-4b-it`.
The v2 adapter's 476 LoRA tensors were remapped into a separate compatible
copy by `scripts/remap_gemma3_adapter.py`; inference verifies every tensor
value after loading. The archived originals are preserved.
