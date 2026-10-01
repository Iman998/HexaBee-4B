"""Independent branded model/dataset cards with truthful release metadata."""

from __future__ import annotations

import html

import yaml

from .evidence import benchmark_markdown, direction_appendix

LANGUAGES = ["fa", "en", "ar", "zh", "he", "es"]


def url(entry: dict) -> str:
    return (
        "https://huggingface.co/"
        + ("datasets/" if entry["repo_type"] == "dataset" else "")
        + entry["repo_id"]
    )


def dataset_loader(entry: dict, configs: list[dict]) -> str:
    name = configs[0]["config_name"]
    split = configs[0]["data_files"][0]["split"]
    return f'''```python
from datasets import load_dataset

ds = load_dataset("{entry["repo_id"]}", "{name}", split="{split}", streaming=True)
print(next(iter(ds)))
```'''


def model_loader(entry: dict) -> str:
    if entry["kind"] == "adapter":
        return f'''```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base = "{entry["base_model"]}"
model = AutoModelForImageTextToText.from_pretrained(
    base, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, "{entry["repo_id"]}")
processor = AutoProcessor.from_pretrained("{entry["repo_id"]}")
model.eval()
```

This repository contains adapter weights. Load them on the **exact base model shown above**. A continued PT→SFT adapter is loaded directly on the original base; do not apply the PT adapter a second time.'''
    return f'''```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

repo = "{entry["repo_id"]}"
processor = AutoProcessor.from_pretrained(repo)
model = AutoModelForImageTextToText.from_pretrained(
    repo, torch_dtype=torch.bfloat16, device_map="auto"
).eval()
messages = [
    {{"role": "system", "content": [{{"type": "text", "text": "Translate the following text from English into Persian."}}]}},
    {{"role": "user", "content": [{{"type": "text", "text": "Language brings people together."}}]}}
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True,
    return_tensors="pt"
).to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The processor must preserve Gemma 3's system/user formatting. This short example uses a new 512-token generation budget; the archived evaluation used its own recorded inference settings.'''


def build_card(entry: dict, all_entries: list[dict], stats: dict, configs: list[dict]) -> str:
    is_model = entry["repo_type"] == "model"
    metadata = {
        "language": entry.get("languages", LANGUAGES),
        "license": entry.get("license", "other"),
        "tags": ["hexabee", "translation", "multilingual", "research", "synthetic-data"],
    }
    if is_model:
        metadata.update(
            library_name="peft" if entry["kind"] == "adapter" else "transformers",
            base_model=entry["base_model"],
            pipeline_tag="image-text-to-text",
        )
        if entry["kind"] == "adapter":
            metadata["tags"].append("lora")
        metadata["license"] = "gemma"
    else:
        metadata["task_categories"] = entry.get("tasks", ["translation"])
        if configs:
            metadata["configs"] = configs
    if metadata["license"] == "other":
        metadata.update(license_name="upstream-terms-apply", license_link="LICENSE.md")
    front = "---\n" + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True) + "---\n\n"
    badges = f"""<p align="center">
<img alt="HexaBee" src="https://img.shields.io/badge/HexaBee-4B-111B2B?style=flat-square&amp;labelColor=B49A62">
<img alt="Artifact" src="https://img.shields.io/badge/{entry["repo_type"]}-release-6BABA8?style=flat-square">
<img alt="Languages" src="https://img.shields.io/badge/languages-{len(metadata["language"])}-B49A62?style=flat-square">
</p>"""
    result_counts = {part["config"]: part["rows"] for part in stats.get("partitions", [])}
    size_fact = (
        ("Checkpoint", "LoRA adapter" if entry["kind"] == "adapter" else "Merged model weights")
        if is_model
        else (
            "Tabular results",
            f"{result_counts['direction_metrics']:,} direction rows + "
            f"{result_counts['paired_judge_scores']:,} paired judge rows",
        )
        if entry["kind"] == "metrics"
        else ("Released rows", f"{stats['rows']:,}")
    )
    facts = [
        ("Artifact", entry["title"]),
        ("Scope", entry["scope"]),
        size_fact,
        (
            "Format",
            "Sharded Parquet + metadata"
            if configs and entry["kind"] != "metrics"
            else "Safetensors + processor"
            if is_model
            else "CSV / JSON + figures",
        ),
        ("Release", "Audited historical artifacts"),
        ("Model family", "HexaBee-4B / Gemma 3"),
    ]
    table = "| Property | Value |\n|:--|:--|\n" + "\n".join(f"| {k} | {v} |" for k, v in facts)
    navigation = "[Overview](#overview) · [Quick start](#quick-start) · [Contents](#contents) · [Provenance](#provenance) · [Limitations](#limitations) · [Release family](#release-family) · [Code & paper](https://github.com/Iman998/HexaBee-4B)"
    family = "| Artifact | Repository |\n|:--|:--|\n" + "\n".join(
        f"| {e['title']} | [{e['repo_id']}]({url(e)}) |" for e in all_entries
    )
    loader = (
        model_loader(entry)
        if is_model
        else dataset_loader(entry, configs)
        if configs
        else f'''```python
from huggingface_hub import snapshot_download
path = snapshot_download("{entry["repo_id"]}", repo_type="dataset")
```'''
    )
    benchmark = benchmark_markdown(stats["benchmark"]) if stats.get("benchmark") else ""
    integrity = "`release_manifest.json` lists distributed files, sizes, and SHA-256. `release_stats.json` records measured artifact metadata."
    if configs and entry["kind"] != "metrics":
        integrity += " Data files use bounded Parquet shards; original strings and empty outputs are preserved in historical exports."
    figures = ""
    if stats.get("benchmark"):
        prefix = (
            "figures/"
            if entry["kind"] == "metrics"
            else "https://huggingface.co/datasets/Iman998/HexaBee-Evaluation/resolve/main/figures/"
        )
        figures = f"![Scores by target language]({prefix}target_comparison.png)\n"
        if entry["kind"] == "metrics":
            figures += f"\n![Direction-level gains over both baselines]({prefix}direction_gains_both.png)\n\n![Judge rubric comparison]({prefix}judge_rubric_profile.png)\n"
    coverage = ""
    if configs:
        coverage = (
            "\n### Available configurations\n\n| Configuration | Splits | Rows |\n|:--|:--|--:|\n"
            + "\n".join(
                f"| `{c['config_name']}` | {', '.join(d['split'] for d in c['data_files'])} | "
                + f"{sum(p['rows'] for p in stats.get('partitions', []) if p['config'] == c['config_name']):,} |"
                for c in configs
            )
        )
    return (
        front
        + f'''<img src="assets/banner.png" alt="{html.escape(entry["title"])} — HexaBee visual identity" width="100%">

# {entry["title"]}

{badges}

<p align="center"><strong>{html.escape(entry["subtitle"])}</strong></p>

{navigation}

## Overview

{entry["description"]}

![Three visual summary cards for {html.escape(entry["title"])}: see the facts and detailed sections below](assets/overview.png)

{table}

{benchmark}

{figures}

## Quick start

{loader}

## Contents

{entry["contents"]}

{integrity}
{coverage}

{direction_appendix(stats["direction_scores"]) if stats.get("direction_scores") else ""}

## Provenance

{entry["provenance"]}

This release is part of **HexaBee-4B**, previously labeled PCT-4B in the local evaluation audit. Both names refer to the same final translator. Renaming does not imply retraining. This repository is an artifact release, not evidence of workshop acceptance or an official shared-task ranking.

## Limitations

{entry["limitations"]}

The complete adaptation recipe was evaluated; no matched ablation isolates the contribution of multilingual answering. Synthetic references and model-based judges limit interpretation. Long COMET inputs are truncated at the encoder limit. Use the independent evaluation repository for precise sample counts, parameters, all directions, and uncertainty estimates.

## Release family

{family}

Each phase, checkpoint, prompt inventory, output archive and evaluation result has its own repository and card. Model cards identify the exact adapter base; dataset cards identify the unit of a released row. Historical row counts are not counts of unique source articles.

## License and attribution

See [LICENSE.md](LICENSE.md). Model weights remain under Gemma terms; upstream dataset and service terms are not replaced by this repository's software license. The illustrated branding is decorative and does not encode experimental evidence.

## Citation and references

**HexaBee: A Six-Language Translator with Multilingual Answering and Bidirectional Supervision.** [Read the submission draft](https://github.com/Iman998/HexaBee-4B/blob/main/paper/HexaBee-4B_WAT2026.pdf) and [browse the reproducibility code](https://github.com/Iman998/HexaBee-4B). Final publication metadata will be added when available.

Related methods and tools: [Gemma 3](https://arxiv.org/abs/2503.19786), [LoRA](https://arxiv.org/abs/2106.09685), [LLaMA-Factory](https://aclanthology.org/2024.acl-demos.38/), [SacreBLEU](https://aclanthology.org/W18-6319/), [COMET](https://aclanthology.org/2020.emnlp-main.213/). Full paper references are distributed with the reproducibility code.

---

<p align="center">⬡ <strong>HexaBee-4B</strong> · FA · EN · AR · ZH · HE · ES<br>Maintained by <a href="https://huggingface.co/Iman998">Iman998</a></p>
'''
    )
