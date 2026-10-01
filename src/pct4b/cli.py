"""One command-line entry point for data construction and evaluation."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from .io import atomic_json, manifest, read_records, write_records


def config(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def parser():
    p = argparse.ArgumentParser(
        prog="pct4b", description="Reproduce PCT-4B data pipelines and evaluation"
    )
    sub = p.add_subparsers(dest="command", required=True)
    streaming = [
        "inspect",
        "normalize-wiki",
        "qa-build",
        "historical-build",
        "export-sft",
        "mt-build",
        "split",
        "judge-sample",
        "archive-import",
        "audit",
        "variants",
        "topics",
        "sources",
        "translate",
        "judge",
    ]
    for name in streaming:
        sp = sub.add_parser(name)
        sp.add_argument("--input", required=True, type=Path)
        sp.add_argument("--output", required=name != "inspect", type=Path)
        sp.add_argument("--limit", type=int, help="Maximum input records; omitted means all")
        if name in ("qa-build", "historical-build"):
            sp.add_argument("--recipe", required=True, type=Path)
        if name in ("qa-build", "mt-build"):
            sp.add_argument(
                "--pools",
                type=Path,
                help="Normalized prompt JSONL; defaults to 36 illustrative seeds",
            )
        if name == "historical-build":
            sp.add_argument("--pool-dir", required=True, type=Path)
            sp.add_argument("--rule-id")
        if name == "export-sft":
            sp.add_argument("--format", choices=["alpaca", "messages"], default="alpaca")
        if name == "mt-build":
            sp.add_argument("--no-reverse", action="store_true")
        if name in ("split", "judge-sample", "mt-build", "translate"):
            sp.add_argument("--seed", type=int, default=42)
        if name == "split":
            sp.add_argument("--test-fraction", type=float, default=0.1)
        if name == "judge-sample":
            sp.add_argument("--per-direction", type=int, default=10)
        if name == "archive-import":
            sp.add_argument("--model", required=True)
            sp.add_argument("--candidate-field", required=True)
        if name in ("variants", "topics", "sources", "translate", "judge"):
            sp.add_argument("--api-config", required=True, type=Path)
            sp.add_argument("--cache", type=Path, default=Path("runs/api.sqlite"))
            sp.add_argument(
                "--execute", action="store_true", help="Enable paid/local chat API calls"
            )
        if name == "variants":
            sp.add_argument("--routes", required=True, type=Path)
        if name == "topics":
            sp.add_argument("--depth", type=int, default=3)
            sp.add_argument("--branching", type=int, default=3)
        if name == "sources":
            sp.add_argument("--config", required=True, type=Path)
        if name == "translate":
            sp.add_argument(
                "--prediction",
                action="store_true",
                help="Produce a model candidate, keeping the reference",
            )
            sp.add_argument("--model-name")
            sp.add_argument(
                "--prompt",
                choices=["teacher", "short", "baseline", "archived-baseline", "archived-short"],
                default="teacher",
            )
    sp = sub.add_parser("prompts")
    sp.add_argument("action", choices=["seeds", "import", "generate"])
    sp.add_argument("--input", type=Path)
    sp.add_argument("--output", required=True, type=Path)
    sp.add_argument("--api-config", type=Path)
    sp.add_argument("--cache", type=Path, default=Path("runs/api.sqlite"))
    sp.add_argument("--execute", action="store_true")
    sp.add_argument("--count", type=int, default=20)
    sp = sub.add_parser("metrics")
    sp.add_argument("--input", type=Path, nargs="+", required=True)
    sp.add_argument("--output", type=Path, required=True)
    sp.add_argument("--metrics", nargs="+", default=["bleu", "chrf", "chrfpp", "ter"])
    sp.add_argument("--checkpoint", type=Path)
    sp.add_argument("--checkpoint-sha256")
    sp.add_argument("--batch-size", type=int, default=32)
    sp.add_argument("--gpus", type=int, default=0)
    sp = sub.add_parser("bootstrap")
    sp.add_argument("--input", type=Path, nargs="+", required=True)
    sp.add_argument("--output", type=Path, required=True)
    sp.add_argument("--system", required=True)
    sp.add_argument("--baselines", nargs="+", required=True)
    sp.add_argument("--replicates", type=int, default=10000)
    sp.add_argument("--seed", type=int, default=20260912)
    sp = sub.add_parser("report")
    sp.add_argument("--input", required=True, type=Path)
    sp.add_argument("--output", required=True, type=Path)
    sp.add_argument("--focus")
    sp.add_argument("--plots", action="store_true")
    return p


def run(args):
    from . import data, evaluation, generation
    from .prompts import PromptPool, generate_paraphrases, import_pools
    from .schema import normalize_wiki

    client = None
    command = args.command
    api = getattr(args, "api_config", None)
    if api:
        if not args.execute:
            print(
                json.dumps(
                    {
                        "status": "dry-run",
                        "command": command,
                        "model": config(api)["model"],
                        "message": "No API calls or outputs written. Add --execute after reviewing input and configuration.",
                    }
                )
            )
            return
        from .api import Client

        args.cache.parent.mkdir(parents=True, exist_ok=True)
        client = Client(config(api), args.cache, execute=True)
    try:
        counts = collections.Counter()
        if command == "prompts":
            if args.action == "seeds":
                rows = PromptPool.load().rows
            elif args.action == "import":
                if not args.input:
                    raise ValueError("prompts import requires --input pool directory")
                rows = import_pools(args.input)
            else:
                if not client or not args.input:
                    raise ValueError(
                        "prompts generate requires --input, --api-config and --execute"
                    )
                rows = generate_paraphrases(client, PromptPool.load(args.input).rows, args.count)
            n = write_records(args.output, rows)
        elif command == "metrics":
            result = evaluation.score_metrics(
                args.input,
                args.metrics,
                args.checkpoint,
                args.batch_size,
                args.gpus,
                args.checkpoint_sha256,
            )
            atomic_json(args.output, result)
            n = len(result["results"])
        elif command == "bootstrap":
            result = evaluation.bootstrap(
                args.input, args.system, args.baselines, args.replicates, args.seed
            )
            atomic_json(args.output, result)
            n = result["n"]
        elif command == "report":
            from .report import build_report

            build_report(config(args.input), args.output, args.focus, args.plots)
            print(f"Report written to {args.output}")
            return
        else:
            if args.limit is not None and args.limit < 1:
                raise ValueError("--limit must be positive")
            rows = read_records(args.input, args.limit)
            if command == "inspect":
                sample = next(rows, {})
                print(
                    json.dumps(
                        {
                            "fields": {k: type(v).__name__ for k, v in sample.items()},
                            "null_fields": [k for k, v in sample.items() if v is None],
                        },
                        indent=2,
                    )
                )
                return
            if command == "normalize-wiki":
                rows = (normalize_wiki(r) for r in rows)
            elif command == "qa-build":
                rows = data.build_qa(rows, PromptPool.load(args.pools), config(args.recipe), counts)
            elif command == "historical-build":
                from .historical import replay

                rows = replay(rows, config(args.recipe), args.pool_dir, counts, args.rule_id)
            elif command == "export-sft":
                rows = data.export_sft(rows, args.format)
            elif command == "mt-build":
                rows = data.build_mt(
                    rows, PromptPool.load(args.pools), not args.no_reverse, args.seed, counts
                )
            elif command == "split":
                counts = data.split_pairs(rows, args.output, args.test_fraction, args.seed)
                print(json.dumps(dict(counts)))
                return
            elif command == "judge-sample":
                rows = data.sample_judge(rows, args.per_direction, args.seed)
            elif command == "archive-import":
                rows = evaluation.import_archive(rows, args.model, args.candidate_field)
            elif command == "audit":
                atomic_json(args.output, data.audit_records(rows))
                return
            elif command == "variants":
                routes = config(args.routes)
                rows = generation.generate_variants(
                    rows, client, routes["routes"], routes.get("fields", ["question", "answer"])
                )
            elif command == "topics":
                rows = generation.expand_topics(rows, client, args.depth, args.branching)
            elif command == "sources":
                rows = generation.generate_sources(rows, client, config(args.config))
            elif command == "translate":
                rows = generation.generate_translations(
                    rows, client, args.seed, args.prompt, args.prediction, args.model_name
                )
            elif command == "judge":
                rows = evaluation.judge(rows, client)
            n = write_records(args.output, rows)
        counts["written"] = n
        inputs = getattr(args, "input", [])
        inputs = inputs if isinstance(inputs, list) else [inputs] if inputs else []
        inputs = [p for p in inputs if p.is_file()]
        # Full file hashes are deliberately skipped for --limit exploratory runs on huge inputs.
        if getattr(args, "limit", None):
            inputs = []
        settings = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k
            not in (
                "input",
                "output",
                "cache",
                "api_config",
                "pool_dir",
                "recipe",
                "pools",
                "checkpoint",
                "routes",
                "config",
            )
            and not isinstance(v, list)
        }
        for field in ("recipe", "pools", "routes", "config", "api_config"):
            value = getattr(args, field, None)
            if value:
                from .io import file_hash

                settings[field + "_sha256"] = file_hash(value)
        if api:
            settings["api"] = {
                k: v
                for k, v in config(api).items()
                if k in ("model", "revision", "temperature", "top_p", "seed", "max_tokens")
            }
            counts["api_calls"] = client.calls
            counts["cache_hits"] = client.cache_hits
        manifest(args.output, command, inputs, settings, dict(counts))
        print(json.dumps(dict(counts), ensure_ascii=False, sort_keys=True))
    finally:
        if client:
            client.close()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, KeyError, RuntimeError, FileNotFoundError) as exc:
        print(f"pct4b: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
