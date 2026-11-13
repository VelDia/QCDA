"""The benchmark's own rules, offline: no model, no network."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import baseline  # noqa: E402
import evaluate as E  # noqa: E402
from common import INVALID, baseline_system, grounded, parse_answer  # noqa: E402


# --------------------------------------------------------------------------- parsing
def test_answer_is_found_inside_prose_and_fences():
    reply = (
        'Here is my answer:\n```json\n{"axis_a": {"case": "not_applicable", '
        '"confidence": "high", "evidence": ""}, "axis_b": {"case": "on_request", '
        '"confidence": "LOW", "evidence": "available on request"}}\n```'
    )
    ans, err = parse_answer(reply)
    assert err == ""
    assert ans["axis_a"]["case"] == "not_applicable"
    assert ans["axis_b"] == {
        "case": "on_request",
        "confidence": "low",
        "evidence": "available on request",
    }


def test_a_near_miss_token_is_invalid_not_repaired():
    ans, err = parse_answer(
        '{"axis_a": {"case": "Openly accessible"}, "axis_b": {"case": "upon_request"}}'
    )
    assert ans["axis_a"]["case"] == INVALID and ans["axis_b"]["case"] == INVALID
    assert "not a case token" in err


def test_no_json_at_all_is_invalid_on_both_axes():
    ans, err = parse_answer("I cannot determine this.")
    assert ans["axis_a"]["case"] == ans["axis_b"]["case"] == INVALID and err


def test_quote_grounding_folds_layout_but_catches_inventions():
    text = "The data are available at https://zenodo.org/record/1 upon request."
    assert grounded("data are available at", text) is True
    assert grounded("deposited in Dryad", text) is False
    assert grounded("", text) is None


def test_both_variants_ask_for_the_same_answer_format():
    for v in ("definitions", "pipeline_prompt"):
        s = baseline_system(v)
        assert '"axis_a": {"case": "<token>"' in s
    assert "DECIDING BETWEEN THE CLOSE PAIRS" in baseline_system("pipeline_prompt")
    assert "DECIDING BETWEEN THE CLOSE PAIRS" not in baseline_system("definitions")


# --------------------------------------------------------------------------- scoring
def _gold(i, a, b, stratum="not_applicable", nda=0):
    return {
        "openalex_id": f"https://openalex.org/W{i}",
        "gold_a": a,
        "gold_b": b,
        "stratum": stratum,
        "nda": nda,
    }


def test_missing_answer_counts_as_wrong_and_flagged():
    rows = E.scored([_gold(1, "not_applicable", "on_request")], {})
    assert rows[0]["missing"] and rows[0]["flagged"] and not rows[0]["both_ok"]


def test_nda_rows_treat_on_request_and_restricted_as_one():
    g = [
        _gold(1, "explicitly_restricted", "on_request", nda=1),
        _gold(2, "explicitly_restricted", "on_request", nda=0),
    ]
    p = {
        x["openalex_id"]: {
            "a": "on_request",
            "b": "explicitly_restricted",
            "a_conf": "high",
            "b_conf": "high",
            "flagged": False,
        }
        for x in g
    }
    r = E.scored(g, p)
    assert r[0]["both_ok"] and not r[1]["a_ok"] and not r[1]["b_ok"]


def test_mcnemar_exact_matches_the_binomial():
    assert E.mcnemar_exact(0, 0) == 1.0
    # b=1, c=9: two-sided exact p = 2 * (C(10,0)+C(10,1)) / 2^10 = 22/1024
    assert abs(E.mcnemar_exact(1, 9) - 22 / 1024) < 1e-12


def test_holm_is_monotone_and_capped():
    assert E.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert max(E.holm([0.9, 0.8])) == 1.0


def test_effort_curve_breaks_ties_in_expectation():
    rows = [
        {"conf": 0, "both_ok": False},
        {"conf": 0, "both_ok": True},
        {"conf": 2, "both_ok": True},
    ]
    c = E.effort_curve(rows)
    assert c == [1.0, 0.5, 0.0, 0.0]


# --------------------------------------------------------------------------- end to end
def test_baseline_and_evaluate_end_to_end(tmp_path):
    texts = tmp_path / "texts"
    texts.mkdir()
    gold = [_gold(i, "not_applicable", "on_request") for i in range(1, 5)]
    for g in gold:
        (texts / f"{g['openalex_id'].rsplit('/', 1)[1]}.txt").write_text(
            "We generated the data, which is available from the authors on request."
        )
    with open(tmp_path / "gold.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gold[0]))
        w.writeheader()
        w.writerows(gold)

    replies = iter(
        [
            # right, grounded
            '{"axis_a":{"case":"not_applicable","confidence":"high","evidence":""},'
            '"axis_b":{"case":"on_request","confidence":"high","evidence":"available from the authors on request"}}',
            # wrong on B, invented quote
            '{"axis_a":{"case":"not_applicable","confidence":"high","evidence":""},'
            '"axis_b":{"case":"openly_accessible","confidence":"high","evidence":"deposited at Zenodo"}}',
            # invalid token
            '{"axis_a":{"case":"none"},"axis_b":{"case":"on_request","confidence":"medium"}}',
            "no idea",
        ]
    )

    def fake(system, user):
        assert "FULL TEXT OF THE PAPER" in user and "on request" in user
        return next(replies), 1000, 50, "end_turn"

    out = tmp_path / "base.jsonl"
    with open(out, "w") as f:
        for g in gold:
            row = baseline.run_one(
                g,
                provider=fake,
                system="s",
                texts=texts,
                max_chars=10_000,
                name="fake",
                model="claude-opus-5-5",
                variant="definitions",
            )
            f.write(json.dumps(row) + "\n")

    # A pipeline that is right on 3 papers and flags the 4th.
    pipe = tmp_path / "pipe"
    pipe.mkdir()
    with open(pipe / "analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "openalex_id",
                "a_case",
                "b_case",
                "a_case_conf",
                "b_case_conf",
                "needs_human_review",
                "model_id",
                "retrieval",
                "text_source",
                "input_tokens",
                "output_tokens",
            ],
        )
        w.writeheader()
        for i, g in enumerate(gold):
            w.writerow(
                {
                    "openalex_id": g["openalex_id"],
                    "a_case": "not_applicable",
                    "b_case": "on_request" if i < 3 else "not_applicable",
                    "a_case_conf": "high",
                    "b_case_conf": "high" if i < 3 else "low",
                    "needs_human_review": i == 3,
                    "model_id": "claude-haiku-5-5",
                    "retrieval": "full_text",
                    "text_source": "arxiv",
                    "input_tokens": 2000,
                    "output_tokens": 100,
                }
            )

    res = tmp_path / "results"
    r = subprocess.run(
        [
            sys.executable,
            str(HERE / "evaluate.py"),
            "--gold",
            str(tmp_path / "gold.csv"),
            "--population",
            "",
            "--texts",
            str(texts),
            "--system",
            f"pipe={pipe / 'analysis.csv'}",
            "--system",
            f"base={out}",
            "--reference",
            "pipe",
            "--out",
            str(res),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    eff = {e["system"]: e for e in csv.DictReader(open(res / "effort.csv"))}
    met = {(m["system"], m["axis"]): m for m in csv.DictReader(open(res / "metrics.csv"))}

    assert float(met[("pipe", "paper")]["accuracy"]) == 0.75
    assert float(met[("base", "paper")]["accuracy"]) == 0.25
    assert int(met[("base", "A")]["invalid"]) == 2  # "none" and "no idea"
    assert int(eff["pipe"]["flagged"]) == 1 and int(eff["pipe"]["silent_errors"]) == 0
    assert float(eff["pipe"]["accuracy_after_review"]) == 1.0
    # baseline: papers 3 and 4 are invalid -> flagged; paper 2 is a silent error
    assert int(eff["base"]["flagged"]) == 2 and int(eff["base"]["silent_errors"]) == 1
    assert float(eff["base"]["quotes_not_in_text"]) == 0.5  # 1 of 2 quotes invented
    # to reach the pipeline's 100% the baseline must review all but its one
    # confident correct paper... its confident wrong one ties with it, so 4.
    assert eff["base"]["reviews_to_reach_reference_accuracy"] == "4"
    # $: pipeline 4 x (2000*0.10 + 100*0.50)/1e6; baseline 4 x (1000*4 + 50*20)/1e6
    assert abs(float(eff["pipe"]["cost_usd"]) - 4 * 250 / 1e6) < 1e-12
    assert abs(float(eff["base"]["cost_usd"]) - 4 * 5000 / 1e6) < 1e-12
    assert (res / "report.md").read_text().startswith("# Benchmark results")
