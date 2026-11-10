"""Score every system against the human gold set: accuracy, agreement, paired
significance, and the manual effort each one leaves behind.

    python evaluate.py \
        --system pipeline=runs/pipeline_claude-opus-5-5/analysis.csv \
        --system asbuilt=predictions/pipeline_asbuilt.csv \
        --system opus_defs=runs/baseline_claude-opus-5-5-definitions.jsonl \
        --system gpt_defs=runs/baseline_gpt-5-definitions.jsonl \
        --reference pipeline --out results

A system is a pipeline export (`analysis.csv`, or any CSV with openalex_id,
a_case, b_case) or a baseline run (`.jsonl` from baseline.py). A gold paper a
system has no valid answer for counts as WRONG for that system, never as
skipped, so a system cannot raise its accuracy by declining hard papers.

ACCURACY AND AGREEMENT, per axis (A external, B self-generated) and for the
paper (both axes right): accuracy with a Wilson 95% interval, Cohen's kappa,
PABAK, macro and weighted F1, and per-class precision/recall/F1. Kappa is
never reported alone (CLAUDE.md conventions). The gold sample is stratified on
the as-built result, so a stratum-reweighted accuracy (to the frame's
population, gold/strata_population.csv) with a stratified-bootstrap interval
is reported beside the sample accuracy.

PAIRED COMPARISON with `--reference`: on the same papers, exact McNemar test
on the discordant pairs, the accuracy difference with a paired bootstrap 95%
interval, and Holm's correction across all comparisons on the same axis.

MANUAL EFFORT. A human reviews the papers a system flags:
  pipeline   needs_human_review (low confidence, no evidence, failed checks)
  baseline   an invalid answer, or `low` confidence on either axis
Reported: papers flagged, the accuracy of what is accepted without review,
the errors that pass unflagged ("silent errors"), and the paper accuracy
after review, assuming a reviewer fixes every paper they are shown. Then two
equal-footing views from each system's own confidence ordering:
  * accuracy after reviewing as many papers as the reference flags;
  * how many reviews it takes to reach the reference's post-review accuracy.
Review time is papers x --minutes-per-check (default 2, the figure the
pipeline's own design assumes; change it to your measured time).
Also: the share of papers whose text the pipeline retrieved with no human
step (baselines were handed that text), quoted evidence that is not in the
paper (a fabricated quote), and API cost from recorded tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import (CONF_RANK, HERE, INVALID, LABELS, agree, grounded, price,
                    read_csv, read_jsonl, wid, wilson, write_csv)

AXES = ("a", "b")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def load_system(name: str, path: Path, texts: Path) -> dict:
    """{openalex_id: {a, b, a_conf, b_conf, flagged, model, tin, tout, ...}} plus meta."""
    preds, meta = {}, {"name": name, "path": str(path), "kind": "", "seconds": None,
                       "grounded": [], "retrieval": Counter(), "usage": []}
    if path.suffix == ".jsonl":
        meta["kind"] = "baseline"
        last = {}
        for r in read_jsonl(path):
            if r.get("ok") or r["openalex_id"] not in last:
                last[r["openalex_id"]] = r
        secs = 0.0
        for oid, r in last.items():
            if not r.get("ok"):
                continue
            p = {"a": r.get("a_case", INVALID), "b": r.get("b_case", INVALID),
                 "a_conf": r.get("a_conf", ""), "b_conf": r.get("b_conf", ""),
                 "model": r.get("model", ""), "truncated": r.get("truncated", False),
                 "has_text": r.get("has_text", False)}
            p["flagged"] = (INVALID in (p["a"], p["b"])
                            or "low" in (p["a_conf"], p["b_conf"]))
            preds[oid] = p
            meta["usage"].append((p["model"], r.get("input_tokens", 0),
                                  r.get("output_tokens", 0)))
            secs += r.get("seconds", 0) or 0
            for ax in AXES:
                g = r.get(f"{ax}_evidence_grounded")
                if g is not None:
                    meta["grounded"].append(bool(g))
        meta["seconds"] = secs
        meta["model_seconds_note"] = "sum of per-request latency"
        return {"preds": preds, "meta": meta}

    meta["kind"] = "pipeline"
    row_usage = []
    for r in read_csv(path):
        row_usage.append((r.get("model_id", ""), int(float(r.get("input_tokens") or 0)),
                          int(float(r.get("output_tokens") or 0))))
        a, b = r.get("a_case") or INVALID, r.get("b_case") or INVALID
        a = a if a in LABELS else INVALID
        b = b if b in LABELS else INVALID
        preds[r["openalex_id"]] = {
            "a": a, "b": b, "a_conf": r.get("a_case_conf", ""),
            "b_conf": r.get("b_case_conf", ""), "model": r.get("model_id", ""),
            "flagged": _truthy(r.get("needs_human_review")) or INVALID in (a, b),
        }
        # A hand-fetched PDF (DAP_PDF_DIR) is text a person had to obtain.
        ret = r.get("retrieval") or "unknown"
        if r.get("text_source") == "local_pdf":
            ret = "hand_fetched"
        meta["retrieval"][ret] += 1
    run = path.parent
    db = run / "dap.sqlite"
    if db.exists():                        # every call, escalations included
        import sqlite3
        con = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
        meta["usage"] = list(con.execute(
            "SELECT model, input_tokens, output_tokens FROM usage"))
    else:
        meta["usage"] = row_usage
    timing = run / "timing.json"
    if timing.exists():
        t = json.loads(timing.read_text())
        meta["seconds"] = sum(t.values())
        meta["timing"] = t
        meta["model_seconds_note"] = "wall clock, all stages"
    ev = run / "evidence.csv"
    if ev.exists() and texts.is_dir():     # independent re-check of the quotes
        for e in read_csv(ev):
            t = texts / f"{wid(e['openalex_id'])}.txt"
            if t.exists() and e.get("quote"):
                meta["grounded"].append(bool(grounded(e["quote"], t.read_text("utf-8"))))
    return {"preds": preds, "meta": meta}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def scored(gold: list[dict], preds: dict) -> list[dict]:
    """One row per gold paper: what was predicted, and whether it is right."""
    out = []
    for g in gold:
        p = preds.get(g["openalex_id"]) or {"a": INVALID, "b": INVALID, "a_conf": "",
                                           "b_conf": "", "flagged": True}
        nda = _truthy(g.get("nda"))
        row = {"id": g["openalex_id"], "stratum": g.get("stratum", ""),
               "flagged": bool(p["flagged"]), "missing": g["openalex_id"] not in preds}
        confs = []
        for ax in AXES:
            gt, pr = g[f"gold_{ax}"], p[ax]
            ok = pr != INVALID and agree(gt, pr, nda=nda)
            row[f"{ax}_gold"], row[f"{ax}_pred"], row[f"{ax}_ok"] = gt, pr, ok
            # On an NDA row the equivalent token is scored as the gold one, so
            # the confusion matrix and the accuracy count the same thing.
            row[f"{ax}_eff"] = gt if ok else pr
            confs.append(-1 if pr == INVALID else CONF_RANK.get(p.get(f"{ax}_conf"), 1))
        row["both_ok"] = row["a_ok"] and row["b_ok"]
        # Confidence used to order reviews: flagged papers first, then lowest.
        row["conf"] = -2 if row["flagged"] else min(confs)
        out.append(row)
    return out


def agreement(rows: list[dict], ax: str) -> dict:
    y = [r[f"{ax}_gold"] for r in rows]
    p = [r[f"{ax}_eff"] for r in rows]
    labels = LABELS + ([INVALID] if INVALID in p else [])
    n = len(y)
    cm = Counter(zip(y, p))
    support = Counter(y)
    predicted = Counter(p)
    po = sum(cm[(l, l)] for l in LABELS) / n
    pe = sum(support[l] * predicted[l] for l in labels) / (n * n)
    k = len(LABELS)
    per_class, f1s, f1w = [], [], []
    for l in LABELS:
        tp = cm[(l, l)]
        s, pr = support[l], predicted[l]
        prec = tp / pr if pr else float("nan")
        rec = tp / s if s else float("nan")
        f1 = 2 * tp / (s + pr) if (s + pr) else float("nan")
        per_class.append({"axis": ax, "case": l, "support": s, "predicted": pr,
                          "precision": prec, "recall": rec, "f1": f1})
        if s + pr:
            f1s.append(f1)
        if s:
            f1w.append((f1, s))
    return {
        "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan"),
        "pabak": (k * po - 1) / (k - 1),
        "macro_f1": sum(f1s) / len(f1s) if f1s else float("nan"),
        "weighted_f1": (sum(f * s for f, s in f1w) / sum(s for _, s in f1w)) if f1w else float("nan"),
        "invalid": predicted[INVALID],
        "per_class": per_class,
        "confusion": cm,
    }


def weighted_accuracy(rows, key, population, *, boot=2000, seed=0):
    """Accuracy reweighted from the stratified sample to the frame population,
    with a stratified bootstrap interval. Strata absent from the sample get no
    weight; the share of the population they hold is reported alongside."""
    by = defaultdict(list)
    for r in rows:
        by[r["stratum"]].append(r[key])
    strata = [s for s in by if s in population]
    if not strata:
        return None
    w = {s: population[s] for s in strata}
    tot = sum(w.values())
    est = sum(w[s] / tot * (sum(by[s]) / len(by[s])) for s in strata)
    rng = random.Random(seed)
    draws = []
    for _ in range(boot):
        v = 0.0
        for s in strata:
            xs = by[s]
            v += w[s] / tot * (sum(rng.choice(xs) for _ in xs) / len(xs))
        draws.append(v)
    draws.sort()
    covered = tot / sum(population.values())
    return est, draws[int(0.025 * boot)], draws[int(0.975 * boot) - 1], covered


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(ref: list[dict], sys_: list[dict], key: str, *, boot=10000, seed=0) -> dict:
    x = [int(r[key]) for r in ref]
    y = [int(s[key]) for s in sys_]
    b = sum(1 for i, j in zip(x, y) if i and not j)     # reference right, system wrong
    c = sum(1 for i, j in zip(x, y) if j and not i)
    n = len(x)
    diff = (sum(y) - sum(x)) / n
    rng = random.Random(seed)
    idx = range(n)
    ds = []
    for _ in range(boot):
        s = [rng.choice(idx) for _ in idx]
        ds.append(sum(y[i] - x[i] for i in s) / n)
    ds.sort()
    return {"diff": diff, "diff_lo": ds[int(0.025 * boot)], "diff_hi": ds[int(0.975 * boot) - 1],
            "ref_only": b, "sys_only": c, "p_mcnemar": mcnemar_exact(b, c)}


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, run = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = run
    return adj


def effort_curve(rows: list[dict]) -> list[float]:
    """Expected paper errors left after reviewing k papers, k = 0..n, reviewing
    the least confident first. Ties are broken in expectation, not by an
    arbitrary order: reviewing j of a tie group of g with e errors removes
    j*e/g of them."""
    groups = defaultdict(lambda: [0, 0])
    for r in rows:
        groups[r["conf"]][0] += 1
        groups[r["conf"]][1] += int(not r["both_ok"])
    errors = sum(e for _, e in groups.values())
    curve = [float(errors)]
    left = float(errors)
    for conf in sorted(groups):
        g, e = groups[conf]
        for _ in range(g):
            left -= e / g
            curve.append(max(left, 0.0))
    return curve


# ---------------------------------------------------------------------------
def fmt(v, pct=False, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{100 * v:.1f}%" if pct else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold", default=str(HERE / "gold" / "frame_review.csv"))
    ap.add_argument("--population", default=str(HERE / "gold" / "strata_population.csv"))
    ap.add_argument("--texts", default=str(HERE / "texts"))
    ap.add_argument("--system", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--reference", help="system the others are compared against")
    ap.add_argument("--minutes-per-check", type=float, default=2.0)
    ap.add_argument("--prices", help='JSON {"model": [in_per_M, out_per_M]} for '
                                     'models not in the built-in table')
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()

    gold = read_csv(a.gold)
    n = len(gold)
    population = {}
    if a.population and Path(a.population).exists():
        population = {r["stratum"]: int(r["population"]) for r in read_csv(a.population)}
    prices = json.loads(Path(a.prices).read_text()) if a.prices else {}
    texts = Path(a.texts)

    systems = {}
    for spec in a.system:
        name, _, path = spec.partition("=")
        systems[name] = load_system(name, Path(path), texts)
    if a.reference and a.reference not in systems:
        sys.exit(f"--reference {a.reference} is not one of the systems")

    rows = {name: scored(gold, s["preds"]) for name, s in systems.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics, per_class, effort, paired_rows = [], [], [], []
    for name, rs in rows.items():
        m = systems[name]["meta"]
        for key, label in (("a_ok", "A"), ("b_ok", "B"), ("both_ok", "paper")):
            k = sum(r[key] for r in rs)
            lo, hi = wilson(k, n)
            row = {"system": name, "kind": m["kind"], "axis": label, "n": n,
                   "correct": k, "accuracy": k / n, "ci_low": lo, "ci_high": hi,
                   "coverage": sum(not r["missing"] for r in rs) / n}
            wa = weighted_accuracy(rs, key, population) if population else None
            if wa:
                row.update(weighted_accuracy=wa[0], weighted_ci_low=wa[1],
                           weighted_ci_high=wa[2], population_covered=wa[3])
            if label != "paper":
                ag = agreement(rs, label.lower())
                row.update({k2: ag[k2] for k2 in ("kappa", "pabak", "macro_f1",
                                                   "weighted_f1", "invalid")})
                per_class += [dict(pc, system=name) for pc in ag["per_class"]]
                write_csv(out / f"confusion_{name}_{label}.csv",
                          [{"gold": g, **{p: ag["confusion"][(g, p)] for p in LABELS + [INVALID]}}
                           for g in LABELS])
            metrics.append(row)

        flagged = sum(r["flagged"] for r in rs)
        auto = [r for r in rs if not r["flagged"]]
        silent = sum(not r["both_ok"] for r in auto)
        curve = effort_curve(rs)
        aurc = sum(curve[k] / (n - k) for k in range(n)) / n
        cost, missing_price = 0.0, set()
        for model, tin, tout in m["usage"]:
            pr = price(model, prices)
            if pr is None:
                missing_price.add(model); continue
            cost += (tin * pr[0] + tout * pr[1]) / 1e6
        g = m["grounded"]
        ret = m["retrieval"]
        auto_text = (ret.get("full_text", 0) + ret.get("partial_text", 0))
        effort.append({
            "system": name, "kind": m["kind"], "n": n,
            "flagged": flagged, "flagged_share": flagged / n,
            "accepted": len(auto),
            "accepted_accuracy": (len(auto) - silent) / len(auto) if auto else float("nan"),
            "silent_errors": silent, "silent_error_share": silent / n,
            "accuracy_after_review": 1 - silent / n,
            "review_minutes": flagged * a.minutes_per_check,
            "review_hours_per_1000_papers": flagged / n * 1000 * a.minutes_per_check / 60,
            "aurc": aurc,
            "quotes_checked": len(g),
            "quotes_not_in_text": (len(g) - sum(g)) / len(g) if g else float("nan"),
            "text_retrieved_automatically": (auto_text / sum(ret.values())) if ret else float("nan"),
            "cost_usd": cost if m["usage"] else float("nan"),
            "cost_usd_per_paper": cost / n if m["usage"] else float("nan"),
            "unpriced_models": ";".join(sorted(missing_price)),
            "seconds": m.get("seconds"), "seconds_note": m.get("model_seconds_note", ""),
            "_curve": curve,
        })

    # Paired comparisons and equal-effort views against the reference.
    if a.reference:
        ref = rows[a.reference]
        ref_eff = next(e for e in effort if e["system"] == a.reference)
        H, target = ref_eff["flagged"], ref_eff["accuracy_after_review"]
        for key, label in (("a_ok", "A"), ("b_ok", "B"), ("both_ok", "paper")):
            group = []
            for name, rs in rows.items():
                if name == a.reference:
                    continue
                group.append(dict(paired(ref, rs, key), system=name, axis=label,
                                  reference=a.reference))
            for r, adj in zip(group, holm([r["p_mcnemar"] for r in group])):
                r["p_holm"] = adj
            paired_rows += group
        for e in effort:
            c = e["_curve"]
            e["accuracy_at_reference_effort"] = 1 - c[min(H, n)] / n
            need = next((k for k, left in enumerate(c) if 1 - left / n >= target - 1e-9), None)
            e["reviews_to_reach_reference_accuracy"] = need if need is not None else "—"

    for e in effort:
        e.pop("_curve")
    write_csv(out / "metrics.csv", metrics)
    write_csv(out / "per_class.csv", per_class)
    write_csv(out / "effort.csv", effort)
    if paired_rows:
        write_csv(out / "paired.csv", paired_rows)
    for name, rs in rows.items():
        write_csv(out / f"predictions_{name}.csv", rs)
    (out / "report.md").write_text(report(a, n, metrics, effort, paired_rows))
    print((out / "report.md").read_text())


def report(a, n, metrics, effort, paired_rows) -> str:
    L = [f"# Benchmark results\n",
         f"Gold: `{a.gold}`, {n} papers, two axes each. A missing or invalid "
         f"answer counts as wrong. Intervals are 95%.\n",
         "## Accuracy and agreement\n",
         "| system | axis | accuracy | 95% CI | reweighted to frame | kappa | PABAK | macro-F1 | invalid |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in metrics:
        wa = (f"{fmt(m.get('weighted_accuracy'), True)} "
              f"[{fmt(m.get('weighted_ci_low'), True)}, {fmt(m.get('weighted_ci_high'), True)}]"
              if "weighted_accuracy" in m else "—")
        L.append(f"| {m['system']} | {m['axis']} | {fmt(m['accuracy'], True)} | "
                 f"[{fmt(m['ci_low'], True)}, {fmt(m['ci_high'], True)}] | {wa} | "
                 f"{fmt(m.get('kappa'))} | {fmt(m.get('pabak'))} | {fmt(m.get('macro_f1'))} | "
                 f"{m.get('invalid', '—')} |")
    if paired_rows:
        L += ["", f"## Paired comparison against `{a.reference}`\n",
              "Difference = system minus reference. McNemar exact test on the "
              "discordant papers; p adjusted by Holm across systems, per axis.\n",
              "| system | axis | difference | 95% CI | ref right only | system right only | p | p (Holm) |",
              "|---|---|---|---|---|---|---|---|"]
        for p in paired_rows:
            L.append(f"| {p['system']} | {p['axis']} | {fmt(p['diff'], True)} | "
                     f"[{fmt(p['diff_lo'], True)}, {fmt(p['diff_hi'], True)}] | "
                     f"{p['ref_only']} | {p['sys_only']} | {fmt(p['p_mcnemar'], nd=4)} | "
                     f"{fmt(p['p_holm'], nd=4)} |")
    L += ["", "## Manual effort\n",
          f"A flagged paper goes to a human ({a.minutes_per_check:g} min each). "
          "Accuracy after review assumes the reviewer fixes every paper shown. "
          "Silent errors are wrong papers nobody is shown.\n",
          "| system | flagged | accepted accuracy | silent errors | accuracy after review | "
          "review h / 1000 papers | AURC | accuracy at ref. effort | reviews to ref. accuracy |",
          "|---|---|---|---|---|---|---|---|---|"]
    for e in effort:
        L.append(f"| {e['system']} | {e['flagged']} ({fmt(e['flagged_share'], True)}) | "
                 f"{fmt(e['accepted_accuracy'], True)} | {e['silent_errors']} | "
                 f"{fmt(e['accuracy_after_review'], True)} | "
                 f"{fmt(e['review_hours_per_1000_papers'], nd=1)} | {fmt(e['aurc'])} | "
                 f"{fmt(e.get('accuracy_at_reference_effort'), True)} | "
                 f"{e.get('reviews_to_reach_reference_accuracy', '—')} |")
    L += ["", "## Evidence, retrieval and cost\n",
          "| system | quotes checked | quotes not in the paper | text found with no human step | "
          "API cost | per paper | time (s) |",
          "|---|---|---|---|---|---|---|"]
    for e in effort:
        L.append(f"| {e['system']} | {e['quotes_checked']} | "
                 f"{fmt(e['quotes_not_in_text'], True)} | "
                 f"{fmt(e['text_retrieved_automatically'], True)} | "
                 f"{'$' + format(e['cost_usd'], '.2f') if not math.isnan(e['cost_usd']) else '—'} | "
                 f"{'$' + format(e['cost_usd_per_paper'], '.4f') if not math.isnan(e['cost_usd_per_paper']) else '—'} | "
                 f"{fmt(e['seconds'], nd=0) if e['seconds'] is not None else '—'} "
                 f"{('(' + e['seconds_note'] + ')') if e['seconds_note'] else ''} |")
    unpriced = {e["system"]: e["unpriced_models"] for e in effort if e["unpriced_models"]}
    if unpriced:
        L.append(f"\nNot costed (no price; pass --prices): {unpriced}")
    L += ["", "Baselines were handed the full text the pipeline retrieved; obtaining "
          "it by hand is effort the baseline numbers do not include."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
