import collections
import json
from pathlib import Path

import pytest

from pct4b.api import Client
from pct4b.data import build_mt, build_qa, split_pairs
from pct4b.evaluation import bootstrap, paired_rows, parse_judge, score_metrics
from pct4b.io import read_records, write_records
from pct4b.prompts import PromptPool, translation_weights
from pct4b.report import build_report, ranked
from pct4b.schema import SECTIONS, normalize_wiki

ROOT = Path(__file__).resolve().parents[1]


def test_json_extension_is_not_format(tmp_path):
    path = tmp_path / "misleading.json"
    path.write_text('{"x":1}\n{"x":2}\n')
    assert list(read_records(path)) == [{"x": 1}, {"x": 2}]
    path.write_text('[{"x":1},{"x":2}]')
    assert len(list(read_records(path))) == 2


def test_atomic_failure_keeps_old_output(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text("previous\n")

    def broken():
        yield {"id": 1}
        raise ValueError("bad record")

    with pytest.raises(ValueError):
        write_records(p, broken())
    assert p.read_text() == "previous\n"


def test_unified_schema_and_missing_variants():
    row = next(read_records(ROOT / "examples/wiki.jsonl"))
    item = normalize_wiki(row)
    assert set(item["variants"]) == {"fa", "en", "ar", "zh", "he", "es"}
    assert "formal_answer" not in item["variants"]["fa"]


def test_qa_is_answering_not_question_translation():
    raw = next(read_records(ROOT / "examples/wiki.jsonl"))
    rule = {"source": "en", "target": "fa", "task": "complete"}
    out = list(build_qa([raw], PromptPool.load(), {"rules": [rule], "seed": 42}))[0]
    assert out["instruct"] == raw["English_instruction"]
    assert out["output"] == raw["Persian_output"]
    assert "context" not in out
    assert out["source_language"] != out["target_language"]


def test_data_reordering_does_not_change_sft():
    rows = list(read_records(ROOT / "examples/wiki.jsonl"))
    rows.append({**rows[0], "id": "different-source", "title": "another"})
    recipe = json.loads((ROOT / "configs/qa_demo.json").read_text())
    pool = PromptPool.load()
    a = {r["id"]: r for r in build_qa(rows, pool, recipe)}
    b = {r["id"]: r for r in build_qa(reversed(rows), pool, recipe)}
    assert a == b


def test_style_missing_is_not_fabricated():
    row = next(read_records(ROOT / "examples/wiki.jsonl"))
    counts = collections.Counter()
    assert not list(
        build_qa(
            [row],
            PromptPool.load(),
            {"rules": [{"source": "en", "target": "fa", "task": "formal"}]},
            counts,
        )
    )
    assert counts["missing_variant"] == 1


def test_all_translation_probabilities():
    for s in ["fa", "en", "ar", "zh", "he", "es"]:
        for t in ["fa", "en", "ar", "zh", "he", "es"]:
            if s == t:
                continue
            w = translation_weights(s, t)
            assert sum(w.values()) == pytest.approx(1)
            assert w[t] == 0.55 and w[s] == 0.30
            assert all(x > 0 for x in w.values())
    assert translation_weights("fa", "en")["zh"] == pytest.approx(0.01)
    # The initial .10 group cap is followed by residual redistribution.
    assert translation_weights("fa", "he")["ar"] == pytest.approx(0.105)


def test_reverse_and_deduplication():
    pair = next(read_records(ROOT / "examples/pairs.jsonl"))
    out = list(build_mt([pair, pair], PromptPool.load()))
    assert len(out) == 2
    assert out[0]["instruct"] == out[1]["output"]
    assert out[0]["output"] == out[1]["instruct"]
    assert out[0]["pair_id"] == out[1]["pair_id"]


def test_split_groups_whitespace_duplicates(tmp_path):
    row = next(read_records(ROOT / "examples/pairs.jsonl"))
    other = {**row, "source": row["source"].replace(" ", "  "), "reference": "different reference"}
    split_pairs([row, other], tmp_path, seed=13)
    tr, te = (
        list(read_records(tmp_path / "train.jsonl")),
        list(read_records(tmp_path / "test.jsonl")),
    )
    assert (len(tr), len(te)) in [(2, 0), (0, 2)]


def judge_value(score=0):
    return {"score": score, "reason": "fixture", "section_scores": dict.fromkeys(SECTIONS, score)}


def test_zero_is_valid_and_bool_is_not_int():
    assert parse_judge("```json\n" + json.dumps(judge_value()) + "\n```")["score"] == 0
    with pytest.raises(ValueError):
        parse_judge(judge_value(True))
    v = judge_value()
    del v["section_scores"]["cohesion"]
    with pytest.raises(ValueError):
        parse_judge(v)


def test_api_cache_and_no_network_by_default(tmp_path):
    calls = []

    def fake(payload):
        calls.append(payload)
        return json.dumps(judge_value())

    cfg = {"model": "test-model", "temperature": 0, "attempts": 1}
    client = Client(cfg, tmp_path / "cache.sqlite", execute=True, transport=fake)
    msg = [{"role": "user", "content": "test"}]
    assert parse_judge(client.complete(msg, parse_judge))["score"] == 0
    client.complete(msg, parse_judge)
    assert len(calls) == 1 and client.cache_hits == 1
    client.close()
    client = Client(cfg, tmp_path / "empty.sqlite", execute=False, transport=fake)
    with pytest.raises(ValueError):
        client.complete(msg)
    client.close()
    assert len(calls) == 1


def test_invalid_api_output_not_cached(tmp_path):
    c = Client({"model": "fake", "attempts": 1}, tmp_path / "cache.sqlite", True, lambda _: "{}")
    with pytest.raises(RuntimeError):
        c.complete([{"role": "user", "content": "x"}], parse_judge)
    assert c.db.execute("SELECT count(*) FROM responses").fetchone()[0] == 0
    c.close()


def test_pairing_rejects_missing_or_duplicate_items(tmp_path):
    rows = list(read_records(ROOT / "examples/demo_a.jsonl"))
    write_records(tmp_path / "partial.jsonl", rows[:-1])
    with pytest.raises(ValueError):
        paired_rows([tmp_path / "partial.jsonl", ROOT / "examples/demo_b.jsonl"])
    write_records(tmp_path / "duplicate.jsonl", rows + [rows[0]])
    with pytest.raises(ValueError):
        paired_rows([tmp_path / "duplicate.jsonl"])


def test_bootstrap_constant_differences():
    out = bootstrap(
        [ROOT / "examples/demo_a.jsonl", ROOT / "examples/demo_b.jsonl"], "demo_b", ["demo_a"], 100
    )
    value = out["paired_differences"]["demo_a"]["reference"]
    assert value["mean"] == 10
    assert value["ci95"] == pytest.approx([10, 10])
    assert value["wins"] == 30


def test_metrics_and_report_are_scoped(tmp_path):
    result = score_metrics([ROOT / "examples/demo_b.jsonl"], ["bleu", "chrf", "chrfpp", "ter"])
    assert len(result["results"]) == 30
    assert result["macro"]["demo_b"]["chrfpp"] == pytest.approx(100)
    assert result["macro"]["demo_b"]["ter"] == 0
    assert "bleu" not in result["pooled"]["demo_b"]
    assert all(r["n"] == 1 for r in result["results"])
    build_report(result, tmp_path, focus="demo_b")
    assert (tmp_path / "tables.tex").exists()
    assert len(list(read_records(ROOT / "examples/demo_b.jsonl"))) == 30


def test_table_ranks_ties_and_ter_direction():
    assert ranked(1, [1, 2, 3], lower=True) == r"\textbf{1.00}"
    assert ranked(2, [1, 2, 3]) == r"\underline{2.00}"
    assert ranked(2.004, [2.001, 2.004, 1]) == r"\textbf{2.00}"


def test_import_archive_keeps_zero():
    from pct4b.evaluation import import_archive

    row = next(read_records(ROOT / "examples/pairs.jsonl"))
    row.update(
        candidate_field=None, judge_reference={"score": None, "raw": json.dumps(judge_value())}
    )
    out = list(import_archive([row], "baseline", "candidate_field"))[0]
    assert out["candidate"] == "" and out["judge_reference"]["score"] == 0


def test_generation_uses_cached_teacher_and_correct_roles(tmp_path):
    from pct4b.generation import generate_translations, generate_variants

    c = Client(
        {"model": "fake", "attempts": 1},
        tmp_path / "cache.sqlite",
        True,
        lambda _: "Translated answer.",
    )
    raw = {"id": 1, "variants": {"fa": {"question": "پرسش", "answer": "پاسخ"}}}
    v = list(generate_variants([raw], c, [["fa", "en"]], ["answer"]))[0]
    assert v["variants"]["fa"]["answer"] == "پاسخ"
    assert v["variants"]["en"]["answer"] == "Translated answer."
    pair = next(read_records(ROOT / "examples/pairs.jsonl"))
    pred = list(generate_translations([pair], c, prediction=True, model_name="fake-system"))[0]
    assert pred["reference"] == pair["reference"] and pred["candidate"] == "Translated answer."
    c.close()
