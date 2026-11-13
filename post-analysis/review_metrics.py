"""Human review vs classifier: export the human codes and score the classifier.

    ../data_availability_pipeline/.venv/bin/python review_metrics.py [human_review.csv] [--out review_results]

The human code is the reference (true), the classifier's stored code is the
prediction. Axis A (external data) and axis B (self-generated data) answer
different questions, so each is scored on its own and they are never combined:
there is no paper-level "result" score, because a combined code can match with
the classifier's value coming from one axis and the human's from the other.

The classifier's stored `result` appears only as the sampling stratum: the
review drew 10 papers per value of it. Rows whose status is not `done`
(e.g. `excluded`) are exported but not scored.

Writes to --out:
    human_review.csv        one row per reviewed paper, human and classifier codes side by side
    review_export.csv       the same papers for reading: title, link, the human review
                            fields, the classifier's codes, and True/False columns
                            axis_a_correct, axis_b_correct and both_correct (A AND B)
    review_export_named.csv the same file with the axes named for readers: axis A is
                            external_data, axis B is article_generated_data
    metrics_summary.csv     per axis: accuracy (+ Wilson 95% CI), macro and weighted P/R/F1, kappa, PABAK
    metrics_per_class.csv   per axis and case: support, predicted, precision, recall, F1
    confusion_a.csv, confusion_b.csv   rows = human, columns = classifier
    per_stratum.csv         per sampling stratum and axis: accuracy with Wilson CI
    misclassified.csv       one row per paper and axis that disagrees, with the human's
                            evidence and notes

NDA rule (Diana, 28 Oct 2026): a statement that mentions a non-disclosure
agreement codes `explicitly_restricted`, but it is a special case of
`on_request`, so for scoring the two count as the same on such a row. A row is
an NDA row when the human's evidence or notes mention one (`nda` column); on it,
a 6-versus-7 disagreement on either axis is scored as agreement.

Because the sample is stratified on the classifier's own output, accuracy,
recall and F1 describe this sample, not the corpus.

--merge FROM=TO scores case FROM as TO on both sides, for a sensitivity run
(e.g. the unruled NDA question: explicitly_restricted=on_request). It drops FROM
from the label set, so kappa and PABAK use one class fewer, and merges the
strata the same way. Write it to its own --out folder.

Undefined values are left blank, never 0: precision for a case the classifier
never predicted, recall for a case the human never assigned. Macro averages
are over the cases where the value is defined; F1 = 2TP / (support + predicted)
is defined for every case that occurs on either side.
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "data_availability_pipeline" / "v2"))
from dap.schema import CASE_NUMBER, Case  # noqa: E402

LABELS = [c.value for c in Case]  # all seven, in publication order
NUM = {c.value: CASE_NUMBER[c] for c in Case}
AXES = (("a", "A (external)"), ("b", "B (self-generated)"))
NDA = re.compile(r"non[- ]?disclosure|\bNDAs?\b", re.I)
NDA_EQUIVALENT = {"on_request", "explicitly_restricted"}

EXPORT_COLS = [
    "openalex_id",
    "doi",
    "title",
    "year",
    "arm",
    "status",
    "stratum",
    "stratum_num",
    "nda",
    "a_case",
    "a_case_num",
    "human_review_a_case",
    "human_review_a_case_num",
    "a_agree",
    "b_case",
    "b_case_num",
    "human_review_b_case",
    "human_review_b_case_num",
    "b_agree",
    "human_review_confidence",
    "human_review_evidence",
    "human_review_notes",
    "human_review_at",
]


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Compute the Wilson score interval for a binomial proportion.

    Args:
        k: The number of successes.
        n: The number of trials.
        z: The normal quantile of the interval; 1.96 gives a 95% interval.

    Returns:
        The `(lower, upper)` bounds, clipped to [0, 1]. Both are NaN when `n`
        is 0.
    """
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Score the classifier against the human codes on one axis.

    Undefined values are NaN, never 0: precision for a case the classifier
    never predicted, recall for a case the human never assigned.

    Args:
        y_true: The human's case token for each paper.
        y_pred: The classifier's case token for each paper, in the same order.
        labels: The case tokens of the scheme, in publication order.

    Returns:
        A tuple `(confusion, per_class, summary)`. `confusion` is the matrix
        with rows = human and columns = classifier. `per_class` has one row
        per case: support, predicted, precision, recall and F1. `summary`
        holds `n`, accuracy with its Wilson interval, macro and weighted
        precision, recall and F1, `kappa`, `pabak`, and the number of classes
        each macro average is taken over.

    Raises:
        ValueError: If either side holds a value that is not in `labels`.
    """
    unknown = (set(y_true) | set(y_pred)) - set(labels)
    if unknown:
        raise ValueError(f"values not in the scheme: {unknown}")
    cm = pd.crosstab(pd.Series(y_true, name="human"), pd.Series(y_pred, name="classifier")).reindex(
        index=labels, columns=labels, fill_value=0
    )
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).to_numpy().astype(float)
    predicted = cm.sum(axis=0).to_numpy().astype(float)
    n = support.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        recall = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(support + predicted > 0, 2 * tp / (support + predicted), np.nan)
    per_class = pd.DataFrame(
        {
            "case": labels,
            "case_num": [NUM[l] for l in labels],
            "support": support.astype(int),
            "predicted": predicted.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

    def weighted(v: np.ndarray) -> float:
        m = ~np.isnan(v) & (support > 0)
        return float(np.average(v[m], weights=support[m])) if m.any() else np.nan

    po = tp.sum() / n
    pe = float((support * predicted).sum() / (n * n))
    k = len(labels)
    lo, hi = wilson(tp.sum(), n)
    summary = {
        "n": int(n),
        "accuracy": po,
        "accuracy_ci_low": lo,
        "accuracy_ci_high": hi,
        "macro_precision": np.nanmean(precision),
        "macro_recall": np.nanmean(recall),
        "macro_f1": np.nanmean(f1),
        "weighted_precision": weighted(precision),
        "weighted_recall": weighted(recall),
        "weighted_f1": weighted(f1),
        "kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan,
        "pabak": (k * po - 1) / (k - 1),
        "classes_in_macro_precision": int((~np.isnan(precision)).sum()),
        "classes_in_macro_recall": int((~np.isnan(recall)).sum()),
        "classes_in_macro_f1": int((~np.isnan(f1)).sum()),
    }
    return cm, per_class, summary


def per_stratum(scored: list[dict], labels: list[str]) -> pd.DataFrame:
    """Compute the accuracy of each axis inside each sampling stratum.

    The axes are scored separately.

    Args:
        scored: The scored review rows; each has `stratum`, `a_agree`,
            `b_agree` and `human_review_confidence`.
        labels: The case tokens that can name a stratum, in publication order.

    Returns:
        One row per stratum that occurs and axis: `n`, `correct`, `accuracy`
        with its Wilson interval, and how many of the stratum's papers the
        human coded with low confidence.
    """
    out = []
    for h in labels:
        grp = [r for r in scored if r["stratum"] == h]
        if not grp:
            continue
        for ax, _ in AXES:
            n, k = len(grp), sum(r[f"{ax}_agree"] for r in grp)
            lo, hi = wilson(k, n)
            out.append(
                {
                    "stratum": h,
                    "stratum_num": NUM[h],
                    "axis": ax,
                    "n": n,
                    "correct": k,
                    "accuracy": k / n,
                    "ci_low": lo,
                    "ci_high": hi,
                    "human_low_confidence": sum(r["human_review_confidence"] == "low" for r in grp),
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    """Export the human review and score the classifier against it.

    Reads the command line (see the module docstring), writes the files it
    lists to `--out` and prints the summary tables.

    Raises:
        SystemExit: If a `--merge` value is not a case token of the scheme.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="?", default=str(HERE / "human_review.csv"))
    ap.add_argument("--out", default=str(HERE / "review_results"))
    ap.add_argument(
        "--merge",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="sensitivity run: score case FROM as case TO on both sides "
        "(e.g. explicitly_restricted=on_request); repeatable. The CSV is not changed.",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    merge = dict(m.split("=", 1) for m in args.merge)
    bad = (set(merge) | set(merge.values())) - set(LABELS)
    if bad:
        raise SystemExit(f"--merge values not in the scheme: {bad}")
    labels = [l for l in LABELS if l not in merge]
    if merge:
        print("merged for scoring: " + ", ".join(f"{k} -> {v}" for k, v in merge.items()))

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    reviewed = [r for r in rows if r.get("human_review_a_case") and r.get("human_review_b_case")]
    for r in reviewed:
        for col in ("a_case", "b_case", "human_review_a_case", "human_review_b_case"):
            if r[col] in merge:
                r[col] = merge[r[col]]
                r[col + "_num"] = str(NUM[r[col]])
        r["stratum"] = merge.get(r["result"], r["result"])
        r["stratum_num"] = str(NUM[r["stratum"]]) if r["stratum"] else ""
        r["nda"] = int(bool(NDA.search(r["human_review_evidence"] + " " + r["human_review_notes"])))
        for ax, _ in AXES:
            clf, hum = r[f"{ax}_case"], r[f"human_review_{ax}_case"]
            nda_equal = r["nda"] and clf != hum and {clf, hum} <= NDA_EQUIVALENT
            r[f"{ax}_nda_equal"] = int(bool(nda_equal))
            r[f"{ax}_agree"] = int(clf == hum or bool(nda_equal))
    pd.DataFrame(reviewed)[EXPORT_COLS].to_csv(out / "human_review.csv", index=False)
    export = pd.DataFrame(
        [
            {
                "openalex_id": r["openalex_id"],
                "title": r["title"],
                "link": f"https://doi.org/{r['doi']}" if r["doi"] else r["openalex_id"],
                "year": r["year"],
                "status": r["status"],
                **{
                    c: r[c]
                    for c in (
                        "human_review_a_case",
                        "human_review_a_case_num",
                        "human_review_b_case",
                        "human_review_b_case_num",
                        "human_review_confidence",
                        "human_review_evidence",
                        "human_review_notes",
                        "human_review_at",
                    )
                },
                "classifier_a_case": r["a_case"],
                "classifier_a_case_num": r["a_case_num"],
                "classifier_b_case": r["b_case"],
                "classifier_b_case_num": r["b_case_num"],
                "classifier_result": r["stratum"],
                "classifier_result_num": r["stratum_num"],
                "axis_a_correct": bool(r["a_agree"]),
                "axis_b_correct": bool(r["b_agree"]),
                "both_correct": bool(r["a_agree"] and r["b_agree"]),
            }
            for r in reviewed
        ]
    )
    export.to_csv(out / "review_export.csv", index=False)
    export.rename(
        columns=lambda c: (
            c.replace("axis_a", "external_data")
            .replace("axis_b", "article_generated_data")
            .replace("_a_case", "_external_data_case")
            .replace("_b_case", "_article_generated_data_case")
        )
    ).to_csv(out / "review_export_named.csv", index=False)

    scored = [r for r in reviewed if r["status"] == "done" and r["a_case"] and r["b_case"]]
    left_out = [r["openalex_id"] for r in reviewed if r not in scored]
    print(
        f"{len(reviewed)} reviewed rows exported; {len(scored)} scored "
        f"(left out: {', '.join(left_out) or 'none'})"
    )
    print(
        f"NDA rows scored: {sum(r['nda'] for r in scored)}; 6/7 pairs counted as agreement: "
        + ", ".join(f"{ax.upper()} {sum(r[f'{ax}_nda_equal'] for r in scored)}" for ax, _ in AXES)
        + "\n"
    )

    summaries, per_classes = [], []
    for ax, name in AXES:
        y_true = [r[f"human_review_{ax}_case"] for r in scored]
        # an NDA-equivalent pair is scored as a hit on the human's case
        y_pred = [
            r[f"human_review_{ax}_case"] if r[f"{ax}_nda_equal"] else r[f"{ax}_case"]
            for r in scored
        ]
        cm, per_class, summary = score(y_true, y_pred, labels)
        cm.to_csv(out / f"confusion_{ax}.csv")
        per_class.insert(0, "axis", ax)
        per_classes.append(per_class)
        summaries.append({"axis": ax, **summary})

        print(f"=== Axis {name}  n={summary['n']}")
        print(
            f"accuracy {summary['accuracy']:.3f}  (95% CI {summary['accuracy_ci_low']:.3f}-"
            f"{summary['accuracy_ci_high']:.3f})   kappa {summary['kappa']:.3f}   PABAK {summary['pabak']:.3f}"
        )
        print(
            f"macro    P {summary['macro_precision']:.3f}  R {summary['macro_recall']:.3f}  "
            f"F1 {summary['macro_f1']:.3f}"
        )
        print(
            f"weighted P {summary['weighted_precision']:.3f}  R {summary['weighted_recall']:.3f}  "
            f"F1 {summary['weighted_f1']:.3f}"
        )
        print(
            per_class.drop(columns="axis").to_string(
                index=False, float_format=lambda v: f"{v:.3f}", na_rep="-"
            )
        )
        print("\nconfusion (rows = human, columns = classifier, by case number)")
        print(cm.rename(index=NUM, columns=NUM).to_string())
        print()

    strata = per_stratum(scored, labels)
    strata.to_csv(out / "per_stratum.csv", index=False)
    print("=== Accuracy per sampling stratum (classifier result), each axis on its own")
    print(
        strata.pivot_table(index=["stratum_num", "stratum", "n"], columns="axis", values="accuracy")
        .round(3)
        .to_string()
    )
    print()

    wrong = []
    for r in scored:
        for ax, _ in AXES:
            if not r[f"{ax}_agree"]:
                wrong.append(
                    {
                        "axis": ax,
                        "stratum": r["stratum"],
                        "stratum_num": r["stratum_num"],
                        "classifier_case": r[f"{ax}_case"],
                        "classifier_num": r[f"{ax}_case_num"],
                        "human_case": r[f"human_review_{ax}_case"],
                        "human_num": r[f"human_review_{ax}_case_num"],
                        **{
                            c: r[c]
                            for c in (
                                "openalex_id",
                                "doi",
                                "title",
                                "human_review_confidence",
                                "human_review_evidence",
                                "human_review_notes",
                            )
                        },
                    }
                )
    wrong = pd.DataFrame(wrong).sort_values(["axis", "classifier_num", "human_num", "openalex_id"])
    wrong.to_csv(out / "misclassified.csv", index=False)
    print(
        f"{len(wrong)} disagreeing paper-axes "
        f"(A {sum(wrong.axis == 'a')}, B {sum(wrong.axis == 'b')}) -> misclassified.csv\n"
    )

    pd.DataFrame(summaries).to_csv(out / "metrics_summary.csv", index=False)
    pd.concat(per_classes).to_csv(out / "metrics_per_class.csv", index=False)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
