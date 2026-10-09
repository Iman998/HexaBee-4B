# Checkpoint ablation on the archived translation test

This evaluation compares three available intermediate checkpoints with the archived
final HexaBee outputs. It uses the same 36,572 synthetic test identities and all
30 directions. The new predictions use the `archived-short` translation prompt,
greedy Transformers generation, and a 5,000-token ceiling. Outputs are written to
`/mnt/ai_team/iman_barati/iman_31/paper/translate/ablation_eval/`.

| Variant | Base | Added weights | Interpretation |
|:--|:--|:--|:--|
| Phase 1 | Gemma 3 4B IT | Wikipedia PT LoRA | Wikipedia continuation only |
| Phases 1+2 | Released multilingual merged base | None | Adds multilingual answering and answer-text translation |
| Phase 3 only | Gemma 3 4B IT | `gemma-3-4b-translate-v2` LoRA | Independent bidirectional translation specialization |
| Phases 1+2+3 | Released final HexaBee outputs | None | Published complete system |

The phase-3-only v2 adapter uses rank 512 and alpha 1024; the final stage-3
adapter uses a different training configuration and is based on the phase-1+2
checkpoint. Therefore v2 is a useful translation-only comparator, but it is
not a controlled removal of the first two stages. Phase 1 → phases 1+2 →
complete system gives the clearest incremental comparison. The archived final
outputs also used a different serving engine, so engine differences remain a
limitation when comparing the new runs with the published final output.

## Adapter compatibility

The archived phase-1 and v2 LoRA files each contain 476 tensors under an older
Gemma module path. The installed model expects a newer path. Run
`scripts/remap_gemma3_adapter.py` in a Slurm CPU allocation to make a separate
copy with mapped keys and exact target modules. The original adapters are
preserved. `scripts/ablation_infer.py` checks every tensor value after loading
and refuses to generate if any are missing or changed.

The original Gemma checkpoint is at `/mnt/ai_team/models/gemma-3-4b-it` on the
cluster. The Slurm wrapper uses this path unless `HEXABEE_GEMMA_BASE` is set.

## Run and score

The cluster wrapper `jobs/ablation_infer.sbatch` selects the input model by its
phase argument. Run each variant through Slurm with job names such as
`eval+phase1`, `eval+phase12`, and `eval+phase3`. Use
`jobs/ablation_metrics.sbatch` after inference has completed. Each prediction
record retains the test ID, source, reference, raw candidate, and inference
settings. A SQLite file checkpoints progress after every batch; a complete
`predictions.jsonl` is exported atomically only after all identities are done.

The non-judge metrics are BLEU, chrF, chrF++, and TER from the same `pct4b`
evaluator used for the release. Scores are computed per direction and averaged
equally over the 30 directions. `metrics.json` also records scorer signatures
and sufficient statistics. Compare systems only after checking all 36,572
identities and reporting output-length limits, empty candidates, and any failed
or resumed Slurm jobs. COMET can be added with the pinned checkpoint described
in [evaluation.md](evaluation.md); it is not part of the default metric job.

The phase-1 smoke test produced unrelated and often 5,000-token-long text on
the first four examples. These are raw predictions, not translation failures
filtered out after generation. The full run may need more than one Slurm time
allocation; resubmit the same wrapper only after the previous job has stopped,
so it resumes its SQLite progress without overlapping writers.
