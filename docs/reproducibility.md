# What can be reproduced

| Workflow | Available in this release | Additional inputs needed |
|:--|:--|:--|
| Offline preparation and evaluation demo | Code, six-language fixtures, mock scores | Python dependencies |
| Numeric tables and figure regeneration | Numeric results, training trace, plotting code, explanatory figures | Metrics/plot dependencies |
| Paired judge intervals | 900 score rows; identity/direction labels; seed | NumPy |
| Full raw-output metric calculation | Pinned metric implementation and CLI | Original source/reference/candidate files; COMET checkpoint for COMET |
| Historical QA/answer-translation construction | 329 policies and retained component counts | Original ordered partitions and exact prompt pools |
| New multilingual/synthetic generation | Configurable cached pipelines | Input data and a configured teacher endpoint |
| New phase-1 PT experiment | Raw-text registration, clearly labeled starter YAML, Slurm job | Raw text, Gemma base, compatible LLaMA-Factory environment, GPUs |
| Stage-2/3 training and merges | Portable YAML, registration/merge scripts, saved final adapter settings | LLaMA-Factory environment, data, preceding checkpoint/adapter, GPUs |
| Bit-identical model reconstruction from public base | Not established | Missing complete stage-1 recipe and immutable historical service/training provenance |

## Historical data versus new construction

The 22 phase-2 datasets combine 329 component exports into 30,153,404 retained records. Their 2,025,730 underlying source rows are reused across tasks/languages. Retained QA includes 5,759,806 cross-language records. Actual complete-answer QA covers 27 of 30 cross-language directions, and 1,961 records have known response-language instruction conflicts. Historical outputs also retain some empty strings. The default new builder validates explicit target directives and skips missing variants, while historical replay keeps flagged inconsistencies. Changing these policies changes the training corpus.

Source partitions and existing translations cannot be recreated exactly from their row counts. Some original translation constructors allowed missing-column fallbacks to a generic text column or an empty column; the public adapters do not fabricate these fallbacks. The recipe field `historical_retained_records` is an audited observation, not a guarantee of output count for an arbitrary new input.

The final synthetic collection contained 365,718 pairs, split historically into 329,146 train and 36,572 test pairs. Nine train pairs were removed before forward/reverse expansion, giving $2(329146-9)=658274$ SFT examples. New deterministic splitting preserves source groups and therefore differs from the original unseeded row split. Original split membership is required for historical score reproduction.

## Interpretation of the reported improvement

Translation is the primary task, with multilingual answering an important component of the proposed recipe. No matched ablation removes only QA while retaining all other data and compute, and no pre-specialization checkpoint comparison or final multilingual-QA benchmark is available. Thus the observed translation gains support the complete system; their size cannot be causally assigned to the QA component alone.

The test sources/references are synthetic. Historical baseline and PCT-4B inference templates differ, including wrappers. The judge is model-based; references and judging share a model family. COMET truncates long inputs at its encoder limit. The training validation subset is very small. The result should not be described as an independent human benchmark, an official shared-task ranking, a prompt-language ablation, or proof of robustness across domains.

## Software evidence

The release includes metric signatures, COMET checkpoint provenance, recipe hashes in newly produced manifests, and a minimal test environment lock. Dependency ranges support installation; the lock records the environment actually used for the release's offline verification. GPU training, hosted model generation, and a fresh full COMET pass are not executed by the unit tests. Stored COMET outputs come from the paper evaluation; testing the wrapper with a mock does not validate the underlying neural metric.
