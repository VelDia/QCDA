"""Shared by the benchmark scripts: paths, the vocabulary, the scoring rules.

The vocabulary is imported from the pipeline's own schema, so the baselines
are scored against exactly the seven tokens the pipeline can emit.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PIPELINE = ROOT / "data_availability_pipeline" / "v2"
sys.path.insert(0, str(PIPELINE))

from dap.classify import KNOWN_PRICES, _norm  # noqa: E402
from dap.schema import CASE_NUMBER, Case  # noqa: E402

LABELS = [c.value for c in Case]
NUM = {c.value: CASE_NUMBER[c] for c in Case}
INVALID = "__invalid__"  # no answer, unparseable answer, or a non-token
CONFIDENCES = ("high", "medium", "low")
CONF_RANK = {"high": 2, "medium": 1, "low": 0}

#: The NDA ruling: a statement
#: that puts data behind a non-disclosure agreement codes
#: `explicitly_restricted`, a special case of `on_request`, so on an NDA row the
#: two count as the same. Applied identically to every system.
NDA = re.compile(r"non[- ]?disclosure|\bNDAs?\b", re.I)
NDA_EQUIVALENT = {"on_request", "explicitly_restricted"}


def wid(openalex_id: str) -> str:
    return (openalex_id or "").rstrip("/").rsplit("/", 1)[-1]


def read_csv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows: list[dict], fields: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_jsonl(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def agree(gold: str, pred: str, *, nda: bool) -> bool:
    if pred == gold:
        return True
    return nda and {gold, pred} <= NDA_EQUIVALENT


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def grounded(quote: str, text: str) -> bool | None:
    """Is a quoted piece of evidence actually in the text the model was given?
    The same normalisation the pipeline's grounding check uses.
    """
    q = _norm(quote or "")
    if not q:
        return None
    return q in _norm(text or "")


# ----------------------------------------------------------------------------
# The baseline prompt: the class definitions and nothing else.
# ----------------------------------------------------------------------------
#: Axis and case definitions as the human reviewers saw them (review_app.py's
#: CASE_HINT, which is the classifier prompt's definitions shortened). This is
#: "an LLM given the classes": no passage selection, no metadata lookup, no
#: decision rules for the close pairs, no validator, no grounding check, no
#: link verification.
DEFINITIONS = """\
Classify a research paper by how it makes its data available. Score two axes
independently; a paper may score on both, either, or neither.

Axis A - EXTERNAL data: data that existed independently of this paper and that
the paper analyses (a pre-collected dataset, real-world records, an established
benchmark, prior artifacts).

Axis B - SELF-GENERATED data: data this paper produced itself by running an
experiment, a simulation, or a generative procedure.

For each axis choose exactly one case token:

  not_applicable            The axis does not apply: no external data analysed
                            (A) / nothing generated (B)
  openly_accessible         A concrete identifier is given: DOI, URL, accession,
                            repository
  access_not_provided       The material is identifiable (e.g. "we use
                            ImageNet") but no path to it is given
  insufficiently_specified  The material cannot even be identified ("an
                            internal corpus")
  access_unresolvable       A specific path is given, and it fails
  on_request                Offered on contact, however hedged
  explicitly_restricted     Stated that the data cannot be shared

Answer with one JSON object and nothing else:

{"axis_a": {"case": "<token>", "confidence": "high|medium|low",
            "evidence": "<a short verbatim quote from the paper, or empty>"},
 "axis_b": {"case": "<token>", "confidence": "high|medium|low",
            "evidence": "<a short verbatim quote from the paper, or empty>"}}
"""


def baseline_system(variant: str) -> str:
    if variant == "definitions":
        return DEFINITIONS
    if variant == "pipeline_prompt":
        # Ablation: the pipeline's full instructions (decision rules for the
        # close pairs, the no-data exception, the evidence rules) but none of
        # its machinery. Separates "a better prompt" from "a pipeline".
        from dap.classify import SYSTEM_PROMPT

        answer_format = DEFINITIONS[DEFINITIONS.index("Answer with one JSON object") :]
        return (
            SYSTEM_PROMPT + "\n\nIN THIS RUN you receive the paper's full text instead of "
            "passages, and no metadata. There is no tool to call.\n\n" + answer_format
        )
    raise ValueError(f"unknown variant {variant!r}")


def parse_answer(text: str) -> tuple[dict, str]:
    """The model's JSON answer, leniently located, strictly validated.

    Returns ({axis_a:{case,confidence,evidence}, axis_b:{...}}, error). A case
    that is not one of the seven tokens is INVALID: it is not mapped to the
    nearest token, because a baseline that needs post-hoc repair is not doing
    the task on its own.
    """
    obj, err = None, "no JSON object in the reply"
    dec = json.JSONDecoder()
    for i, ch in enumerate(text or ""):
        if ch != "{":
            continue
        try:
            cand, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and ("axis_a" in cand or "axis_b" in cand):
            obj, err = cand, ""
            break
    out, errors = {}, [err] if err else []
    for ax in ("axis_a", "axis_b"):
        raw = (obj or {}).get(ax) or {}
        if not isinstance(raw, dict):
            raw = {"case": raw}
        case = str(raw.get("case", "")).strip()
        conf = str(raw.get("confidence", "")).strip().lower()
        if obj is not None and case not in LABELS:
            errors.append(f"{ax}: {case!r} is not a case token")
        out[ax] = {
            "case": case if case in LABELS else INVALID,
            "confidence": conf if conf in CONFIDENCES else "",
            "evidence": str(raw.get("evidence") or ""),
        }
    return out, "; ".join(e for e in errors if e)


def price(model: str, prices: dict | None = None) -> tuple[float, float] | None:
    table = dict(KNOWN_PRICES)
    table.update(prices or {})
    return table.get(model)
