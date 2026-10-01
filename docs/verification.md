# Release verification

Verified locally on 2026-09-12 with CPython 3.11.7; exact installed package versions are recorded in `requirements-tested.txt`.

- 34 offline tests passed, including strict identity pairing, missing/duplicate records, valid zero scores, retry/cache behavior, placeholder preservation, source-group splitting, historical replay policies, mock COMET scaling/truncation, CLI execution, and training-config rendering.
- Ruff lint and formatting checks passed.
- Every command in the README offline example completed successfully. Wheel installation loaded both the CLI and all 36 packaged prompt seeds outside the source checkout.
- Paper reconstruction verified 90 system/direction rows, recomputed 270 metric values from released sufficient statistics, checked all 329 data component counts, and reproduced the four paired mean-difference intervals.
- The new archive importer and bootstrap were run on all 900 original judge output rows. Their means and intervals matched the paper. Original texts remain outside the release.
- A 100-row sample from the original Wikipedia export produced 2,866 SFT records with the default scalable recipe. Missing variants were counted and skipped. This smoke-test count is not a claim about the full historical mixture.
- All 341,142 non-hidden original prompt rows were imported, with six component types and direction-conditioned translation metadata. A historical QA rule also ran on the 100-row sample.
- All eight figure families were regenerated as PDF/SVG/PNG. Combined full-test and judge tables compiled into an eight-page preview with no overfull boxes.
- The release ZIP is built from an explicit allowlist with a per-file SHA-256 manifest. Virtual environments, source datasets, model weights, raw prediction/judge texts, API caches, and local runs are excluded.

The local checks do not include hosted teacher generation, GPU training, or a fresh neural COMET pass. Those pathways require the documented external inputs and environments. The COMET wrapper is covered by a mocked integration test; the shipped neural scores are the previously computed paper artifacts.
