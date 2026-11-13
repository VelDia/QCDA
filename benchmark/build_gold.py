"""Build the gold set (and the as-built pipeline predictions) from the human review.

    python build_gold.py
    python build_gold.py --review ../post-analysis/human_review.csv \
        --db ../dap_final.sqlite --out gold

Writes:
  gold/frame_review.csv       one row per human-reviewed paper: the human's two
                              codes (gold_a, gold_b), the sampling stratum and
                              the NDA marker. This is the reference.
  gold/strata_population.csv  how many papers of the frame fall in each stratum,
                              to reweight sample metrics to the frame.
  predictions/pipeline_asbuilt.csv
                              the codes the corpus actually carries for those
                              papers, read from the database: the pipeline as it
                              was run for the case study.
  gold/input_papers.csv       the gold papers as an `enrol` input, so the
                              pipeline can be re-run on them from scratch.

The review was blind: review_app.py never sends the classifier's codes to the
page. The sample is stratified on the as-built `result` (10 papers per value),
so unweighted metrics describe the sample, not the corpus; evaluate.py reports
both.

Rows the pipeline has since excluded (duplicates, correction notices) are left
out of the gold set and listed on stderr.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from common import HERE, LABELS, NDA, ROOT, read_csv, write_csv

sys.path.insert(0, str(ROOT / "post-analysis"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--review", default=str(ROOT / "post-analysis" / "human_review.csv"))
    ap.add_argument("--db", default=str(ROOT / "dap_final.sqlite"))
    ap.add_argument("--out", default=str(HERE / "gold"))
    ap.add_argument("--pred-out", default=str(HERE / "predictions"))
    a = ap.parse_args()

    rows = read_csv(a.review)
    gold, dropped = [], []
    for r in rows:
        ga, gb = r.get("human_review_a_case", ""), r.get("human_review_b_case", "")
        if not ga and not gb:
            continue  # not reviewed
        if r.get("status") == "excluded":
            dropped.append(r["openalex_id"])
            continue
        for v in (ga, gb):
            if v not in LABELS:
                sys.exit(f"{r['openalex_id']}: human code {v!r} is not a case token")
        text = f"{r.get('human_review_evidence', '')} {r.get('human_review_notes', '')}"
        gold.append(
            {
                "openalex_id": r["openalex_id"],
                "doi": r.get("doi", ""),
                "title": r.get("title", ""),
                "publication_year": r.get("year", ""),
                "gold_a": ga,
                "gold_b": gb,
                "stratum": r.get("result") or "blank",
                "nda": int(bool(NDA.search(text))),
                "human_confidence": r.get("human_review_confidence", ""),
                "source": "frame_review",
            }
        )
    out = Path(a.out)
    write_csv(out / "frame_review.csv", gold)
    write_csv(
        out / "input_papers.csv",
        [
            {
                "openalex_id": g["openalex_id"],
                "doi": g["doi"],
                "publication_year": g["publication_year"],
            }
            for g in gold
        ],
    )
    print(f"gold: {len(gold)} papers -> {out / 'frame_review.csv'}", file=sys.stderr)
    if dropped:
        print(f"left out {len(dropped)} excluded: {', '.join(dropped)}", file=sys.stderr)
    print(f"strata: {dict(Counter(g['stratum'] for g in gold))}", file=sys.stderr)

    # As-built predictions and the stratum population, from the database.
    db = Path(a.db)
    if not db.exists():
        print(f"{db} not found: skipping as-built predictions and population", file=sys.stderr)
        return
    from frame_result import add_result, load_papers

    frame = add_result(load_papers(db))
    pop = frame["result"].replace("", "blank").value_counts()
    write_csv(
        out / "strata_population.csv",
        [{"stratum": k, "population": int(v)} for k, v in pop.items()],
    )

    con = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    ids = [g["openalex_id"] for g in gold]
    q = ",".join("?" * len(ids))
    pred = []
    for oid, status, rj in con.execute(
        f"SELECT openalex_id, status, record_json FROM papers WHERE openalex_id IN ({q})", ids
    ):
        rec = json.loads(rj) if rj else {}
        if not rec.get("a_case"):
            continue  # never classified: no prediction
        pred.append(
            {
                k: rec.get(k, "")
                for k in (
                    "openalex_id",
                    "a_case",
                    "a_case_conf",
                    "b_case",
                    "b_case_conf",
                    "needs_human_review",
                    "model_id",
                    "prompt_version",
                    "scheme_version",
                    "retrieval",
                    "text_source",
                    "input_tokens",
                    "output_tokens",
                )
            }
        )
    write_csv(Path(a.pred_out) / "pipeline_asbuilt.csv", pred)
    print(
        f"as-built predictions: {len(pred)} of {len(gold)} "
        f"({Counter(p['model_id'] for p in pred).most_common(3)})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
