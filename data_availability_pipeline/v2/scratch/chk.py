"""Pre-flight: run each record through the REAL schema gate, offline.

Three rejected lines retire a paper for good -- `store.fail` bumps `attempts`,
and at MAX_ATTEMPTS the row is no longer claimable, with no classification to
requeue from. So every results file goes through `_assemble` here, against the
same grounding text, before it is allowed near the store.

Grounding comes from the DATABASE, not the prompt file. The store grounds a
quote against `grounding_text(passages, das_texts)` only -- the title, venue
and year header that the prompt also carries is NOT quotable. An earlier
version of this script normalised the whole prompt, which let a quote taken
from the title pass here and be rejected by the store; that cost `W4409360769`
a retrieval attempt in batch 89. The prompt file's `meta` does not carry
`passages`, so the database is the only place the exact text lives.

A second, non-blocking check: `review_reason` must not carry a count or an
ordinal that ranges over other rows ("the fourth in the corpus", "the five P4
losses in this batch"). Those are written before the batch is finished, they
have been wrong twice -- batch 93's `W4410249248` and batch 94's
`W4410553411` -- and a stored note cannot be corrected by an append, so the
error outlives the batch. Counts belong in `review_issues.md`, where they are
measured after the fact. This prints COUNT-CLAIM and does not fail the file.

A third, non-blocking check: axis A coded `access_not_provided` with no source
named. External input that is never named is `insufficiently_specified` -- the
reader cannot identify the material at all (Diana, 3 Oct 2026, review_issues.md
S83). From 24 Sep to 3 Oct a builder coded it `access_not_provided`, first as
`needs_data` in h.py and then as a `noaxa` that every batch builder from b73 to
b94 redefined locally, which is why this is checked on the record and not on
the builder. `declared=restricted` is exempt: a bare refusal with no reason
lands on `access_not_provided` under P9. This prints UNNAMED-ANP and does not
fail the file.

A fourth, non-blocking check: a record that says `das_present=False` while the
row's own passages carry a statement-shaped heading followed by statement
language. 40 live rows are in that state (S88), and `das_present` is the one
field in the record taken straight from the coder with no validator -- it
carries the study's "N% lack a formal statement" claim. This prints DAS-MISSED
and does not fail the file; the publishers' own disclaimer sentence is
excluded, and an offer made in running text with no heading is not matched at
all, because whether that counts as a formal statement is Diana's to decide.
"""
import json, re, sqlite3, sys
sys.path.insert(0, '.')
from dap.pipeline import _assemble, _summary
from dap.classify import Usage, _norm, grounding_text
from dap.schema import Arm, Retrieval, TextSource, SchemaError

DB = sys.argv[3] if len(sys.argv) > 3 else "../../dap_final.sqlite"

prompts, metas = {}, {}
for line in open(sys.argv[1]):
    o = json.loads(line)
    if o.get("_header"):
        continue
    prompts[o["custom_id"]] = o["prompt"]
    metas[o["custom_id"]] = o["meta"]

db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row


def db_meta(oid):
    r = db.execute("select meta_json from papers where openalex_id=?",
                   (oid,)).fetchone()
    return json.loads(r["meta_json"] or "{}") if r else {}


def header_stripped(cid):
    """Fallback when the row is not in this database: drop the metadata header,
    which is the part the store will not let a quote come from."""
    t = prompts[cid]
    i = t.find("PASSAGES FROM THE PAPER")
    return _norm(t[i:] if i >= 0 else t)


#: An ordinal or a cardinal immediately in front of a phrase that scopes over
#: other rows. Deliberately narrow: "the second both-axes-open row in this
#: batch" trips it, "two passage quotes" does not.
COUNT_CLAIM = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|\d+(?:st|nd|rd|th)?)\b"
    r"[^.;]{0,60}?\b(?:in (?:this|the) (?:batch|corpus)|times in|"
    r"in \w+ batches|(?:P\d|row|rows|batches) (?:in|of) )", re.I)

#: A statement-shaped heading, and the language that follows a real one.
DAS_HEAD = re.compile(r"(data|code|materials?)\s+availability"
                      r"|availability\s+of\s+data|data\s+sharing\s+statement"
                      r"|no\s+associated\s+data", re.I)
DAS_BODY = re.compile(r"(available|not applicable|no associated"
                      r"|on (reasonable )?request|upon request|will be made"
                      r"|included in (this|the) (article|manuscript|paper)"
                      r"|repository|deposited|no data (were|was|are))", re.I)
#: The publisher's own liability sentence, which is not the paper's statement.
DAS_DISCLAIMER = re.compile(r"availability statements?,?\s*or supplementary materials",
                            re.I)


def das_in_text(dm):
    """True when this row's own retrieved text carries a statement."""
    blob = " ".join((p.get("text", "") if isinstance(p, dict) else str(p))
                    for p in (dm.get("passages") or []))
    blob += " " + " ".join(dm.get("das_texts") or [])
    for m in DAS_HEAD.finditer(blob):
        if DAS_DISCLAIMER.search(blob[max(0, m.start() - 60):m.end() + 60]):
            continue
        if DAS_BODY.search(blob[m.end():m.end() + 180]):
            return m.group(0)
    return None


bad = 0
for line in open(sys.argv[2]):
    o = json.loads(line)
    cid, m = o["custom_id"], metas.get(o["custom_id"])
    if m is None:
        print(f"{cid}: NOT IN PROMPT FILE"); bad += 1; continue
    dm = db_meta(m["openalex_id"])
    passages = dm.get("passages") or []
    if passages or dm.get("das_texts"):
        ground = grounding_text(crossref_summary=_summary(dm),
                                passages=passages)
    else:
        print(f"{cid}: WARNING — no passages in {DB}, falling back to the "
              f"header-stripped prompt")
        ground = header_stripped(cid)
    try:
        _assemble(o["classification"],
                  openalex_id=m["openalex_id"], doi=m.get("doi", ""),
                  title=m.get("title", ""), venue=m.get("venue", ""),
                  year=m.get("year"), arm=Arm.coerce(m.get("arm", "other_cs")),
                  model="manual:preflight",
                  retrieval=Retrieval.coerce(m.get("retrieval", "metadata_only")),
                  text_source=TextSource.coerce(m.get("text_source", "none")),
                  text_url=m.get("text_url", ""), usage=Usage(model="manual"),
                  grounding=ground,
                  passages=passages,
                  crossref_das=bool(dm.get("das_texts")))
    except (SchemaError, KeyError, TypeError, ValueError) as e:
        print(f"{cid}: {e}"); bad += 1
    why = o["classification"].get("review_reason") or ""
    for m in COUNT_CLAIM.finditer(why):
        print(f"{cid}: COUNT-CLAIM in review_reason -- "
              f"{m.group(0).strip()!r}; measure it in review_issues.md "
              f"instead, a stored note cannot be fixed by an append")
    axa = o["classification"].get("axis_a") or {}
    if ((axa.get("case") or {}).get("value") == "access_not_provided"
            and not any(str(n).strip() for n in axa.get("names") or [])
            and (axa.get("declared") or {}).get("value") != "restricted"):
        print(f"{cid}: UNNAMED-ANP -- axis A is access_not_provided but names "
              f"no source; unnamed external input is insufficiently_specified "
              f"(S83). Use h.noaxa / h.needs_data, or record the name")
    if not o["classification"].get("das_present"):
        head = das_in_text(dm)
        if head:
            print(f"{cid}: DAS-MISSED -- das_present is false but the row's "
                  f"passages carry {head!r} followed by statement language "
                  f"(S88); read it before coding the row as having none")
print("FAILURES:", bad)
