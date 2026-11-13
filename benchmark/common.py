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
    """Return the bare work id from an OpenAlex identifier.

    Args:
        openalex_id: An OpenAlex id, either bare (`W123`) or as a URL
            (`https://openalex.org/W123`). Empty or None gives an empty string.

    Returns:
        The last path segment of the id, e.g. `W123`.
    """
    return (openalex_id or "").rstrip("/").rsplit("/", 1)[-1]


def read_csv(path: str | Path) -> list[dict]:
    """Read a UTF-8 CSV file into a list of rows.

    Args:
        path: The CSV file to read. Its first line is the header.

    Returns:
        One dict per data row, keyed by column name.
    """
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict], fields: list[str] | None = None) -> None:
    """Write rows to a UTF-8 CSV file, creating parent directories as needed.

    Args:
        path: The file to write. An existing file is overwritten.
        rows: The rows to write, one dict per row.
        fields: The columns to write, in order. Defaults to the keys of the
            first row; keys of a row that are not in `fields` are dropped.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSON Lines file, skipping blank lines.

    Args:
        path: The `.jsonl` file to read, UTF-8.

    Returns:
        One decoded object per non-blank line, in file order.
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def agree(gold: str, pred: str, *, nda: bool) -> bool:
    """Decide whether a predicted case counts as agreeing with the gold case.

    Args:
        gold: The human reviewer's case token.
        pred: The system's case token.
        nda: Whether the paper is an NDA row. On such a row `on_request` and
            `explicitly_restricted` count as the same answer (see `NDA`).

    Returns:
        True if the tokens are equal, or if `nda` is set and both tokens are
        in `NDA_EQUIVALENT`.
    """
    if pred == gold:
        return True
    return nda and {gold, pred} <= NDA_EQUIVALENT


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
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
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def grounded(quote: str, text: str) -> bool | None:
    """Check that a quoted piece of evidence is in the text the model was given.

    Uses the same normalisation as the pipeline's grounding check.

    Args:
        quote: The evidence the model quoted. May be empty or None.
        text: The text the model was given. May be empty or None.

    Returns:
        True if the normalised quote occurs in the normalised text, False if
        it does not, and None if there is no quote to check.
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
    """Build the system prompt for a baseline variant.

    Args:
        variant: `"definitions"` for the class definitions and the answer
            format only, or `"pipeline_prompt"` for the pipeline's full system
            prompt followed by the same answer format (the ablation).

    Returns:
        The system prompt text.

    Raises:
        ValueError: If `variant` is not one of the two known variants.
    """
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
    """Locate the model's JSON answer leniently and validate it strictly.

    A case that is not one of the seven tokens is INVALID: it is not mapped to
    the nearest token, because a baseline that needs post-hoc repair is not
    doing the task on its own.

    Args:
        text: The model's whole reply. The answer may be surrounded by prose
            or code fences; the first JSON object that has an `axis_a` or
            `axis_b` key is taken.

    Returns:
        A pair `(answer, error)`. `answer` is
        `{"axis_a": {"case", "confidence", "evidence"}, "axis_b": {...}}`,
        where `case` is a case token or `INVALID` and `confidence` is one of
        `CONFIDENCES` or empty. `error` describes every problem found, joined
        with `"; "`, and is empty when the answer is valid.
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
    """Look up the price of a model.

    Args:
        model: The model id, as recorded in a run.
        prices: Extra or overriding prices, as `{model: (input, output)}`.
            They take precedence over the pipeline's `KNOWN_PRICES`.

    Returns:
        The `(input, output)` price in USD per million tokens, or None if the
        model is in neither table.
    """
    table = dict(KNOWN_PRICES)
    table.update(prices or {})
    return table.get(model)
