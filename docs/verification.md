# Verification record

On 2026-10-02, the source archive was extracted in the user's Maxwell project directory and checked inside a bounded `general` Slurm CPU allocation. The validation environment used Python 3.12 with `.[metrics,plots,hub,dev]` installed from the internal Nexus mirror.

| Check | Result |
|:--|:--|
| Ruff lint | Passed |
| Ruff formatting | 40 files formatted |
| Offline pytest suite | 59 passed |
| `run_workflow.py --stage all` | Completed preparation, five trainer/export configs, metrics, bootstrap, report, and numeric verification |
| Archived evidence check | 90 system/direction rows and 329 construction components verified |
| Figure generation from code | Eight versioned figure families regenerated on demand |
| Source package | 162 files, including 45 explanatory visuals; no manuscript or source archive |

The workflow demo produced 43 phase-2 SFT rows and 50 forward/reverse translation rows from deliberately tiny fixtures. Its scores are artificial. GPU training, hosted teacher generation, inference against the final model, and a fresh COMET pass were not part of this offline check; their inputs and environment requirements are documented in the stage guides.

To repeat the checks:

```bash
sbatch jobs/workflow.sbatch configs/workflow.demo.json all
sbatch jobs/cli.sbatch pytest -q
sbatch jobs/cli.sbatch ruff check src scripts tests
sbatch jobs/cli.sbatch ruff format --check src scripts tests
```
