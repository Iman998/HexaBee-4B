"""Topic expansion, native synthetic texts, aligned translations and wiki variants."""

import json

from .api import parse_json
from .io import digest
from .schema import LANGUAGES, language, normalize_wiki

LONG_TRANSLATION = """You are an expert translator. Translate the following text from {source} into {target}.
Output ONLY the translated text: no explanations, headings, language names, tags, quotes, code fences, or metadata.
Preserve formatting, line breaks, paragraphs, lists, tables, numbers, dates, units, names, trademarks and acronyms.
Preserve markup and code; translate only visible prose. Match the source tone and register.
Do not add or omit content. Use natural wording in the target language."""


def translation_messages(source, target, value, prompt="teacher"):
    s, t = LANGUAGES[language(source)], LANGUAGES[language(target)]
    if prompt.startswith("archived-"):
        from .archived_prompts import Prompts

        if prompt == "archived-short":
            return Prompts.text_translation_messages_short(value, s, t, "gemma-3-4b-translate")
        return Prompts.text_translation_messages(value, s, t)
    if prompt == "short":
        system = f"Translate the following text from {s} into {t}."
        user = "\n" + value
    else:
        system = LONG_TRANSLATION.format(source=s, target=t)
        user = "Text:\n" + ("'''" + value + "'''" if prompt == "baseline" else value) + "\n"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_variants(rows, client, routes, fields=("question", "answer")):
    for raw in rows:
        row = normalize_wiki(raw)
        for s, t in routes:
            for field in fields:
                # Only fill missing target fields; preserve existing translations.
                if row["variants"].get(t, {}).get(field):
                    continue
                source = row["variants"].get(s, {}).get(field)
                if not source:
                    continue
                answer = client.complete(translation_messages(s, t, source))
                row["variants"].setdefault(t, {})[field] = answer
        row["variant_teacher"] = {
            "model": client.config["model"],
            "revision": client.config.get("revision"),
        }
        yield row


def expand_topics(rows, client, depth=3, branching=3):
    if depth < 1 or branching < 1:
        raise ValueError("Depth and branching must be positive")

    def validate(raw):
        values = parse_json(raw)
        if not isinstance(values, list) or not all(
            isinstance(x, str) and x.strip() for x in values
        ):
            raise ValueError("Expected topic-name strings")
        if len(set(values)) != len(values) or len(values) not in (0, branching):
            raise ValueError("Expected distinct children, or [] for a leaf")
        return values

    for row in rows:
        stack = [row.get("topic_path", [row.get("topic", "")])]
        if not stack[0] or not all(stack[0]):
            raise ValueError("Topic input needs topic_path or topic")
        while stack:
            path = stack.pop()
            if len(path) >= depth:
                yield {"topic_path": path, "id": digest(path)}
                continue
            raw = client.complete(
                [
                    {
                        "role": "system",
                        "content": "Build a taxonomy. Return a JSON array of exactly the requested number of unique recognized subdomains ONE level below the given path. Use concise noun phrases and minimal overlap. Return [] if this is a leaf. No extra text.",
                    },
                    {"role": "user", "content": json.dumps({"path": path, "count": branching})},
                ],
                validate,
            )
            children = validate(raw)
            if not children:
                yield {"topic_path": path, "id": digest(path)}
            stack.extend(path + [child] for child in reversed(children))


def generate_sources(rows, client, config):
    import numpy as np

    for row in rows:
        path = row["topic_path"]
        for lang in config.get("languages", list(LANGUAGES)):
            for replica in range(config.get("per_topic_language", 1)):
                key = digest([path, lang, replica, config.get("seed", 42)])
                rng = np.random.default_rng(int(key[:16], 16))
                attrs = {k: str(rng.choice(v)) for k, v in config["attributes"].items()}
                req = {"topic_path": path, "language": LANGUAGES[lang], **attrs}
                raw = client.complete(
                    [
                        {
                            "role": "system",
                            "content": "Write idiomatic native text for the given topic and language, following all supplied length, genre, register, complexity and noise attributes. Generate directly in that language, not a translation. Return only the text. Avoid personal identifiers and copied passages.",
                        },
                        {"role": "user", "content": json.dumps(req, ensure_ascii=False)},
                    ]
                )
                yield {
                    "id": key,
                    "topic_path": path,
                    "source_language": lang,
                    "source": raw,
                    "attributes": attrs,
                    "generation": {
                        k: client.config.get(k)
                        for k in ("model", "revision", "temperature", "top_p", "seed")
                    },
                }


def generate_translations(
    rows, client, seed=42, prompt="teacher", prediction=False, model_name=None
):
    for raw in rows:
        row = dict(raw)
        s = language(row.get("source_language", row.get("src_language")))
        source = row.get("source", row.get("text"))
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Missing source text")
        t = row.get("target_language", row.get("tgt_lang"))
        if t:
            t = language(t)
        else:
            choices = [x for x in LANGUAGES if x != s]
            # Reproducible uniform non-self selection for new runs.
            import random

            t = random.Random(digest([seed, row.get("id"), s, source])).choice(choices)
        if s == t:
            raise ValueError("Source and target languages must differ")
        answer = client.complete(translation_messages(s, t, source, prompt))
        result = {
            **row,
            "source_language": s,
            "target_language": t,
            "source": source,
            "candidate" if prediction else "reference": answer,
            "inference": {
                "prompt": prompt,
                **{
                    k: client.config.get(k)
                    for k in ("model", "revision", "temperature", "max_tokens")
                },
            },
        }
        if prediction:
            result["model"] = model_name or client.config["model"]
        from .schema import normalize_pair

        yield normalize_pair(result)
