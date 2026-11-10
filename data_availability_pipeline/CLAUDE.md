# Data-availability study — working brief

Read this before doing anything. It is the standing context for this repo.

## What this is

A bibliometric study of how research papers make their data available. Papers are
coded on two independent axes against a fixed seven-case scheme; the coded corpus
supports claims about how often data is reachable, and what lies behind a missing
availability statement.

A prototype (v1) exists and is **not** compatible with the current scheme. Its
axes are inverted and its case vocabulary is different. See "v1" below.

## Status

**Decided.** Two axes, seven cases, tokens as defined in `v2/dap/schema.py`, scheme 2.0.1.
Version of record is the unit; preprint only where no published version exists.
Window 2021–2026, whole calendar years — narrowed from 2017–2026 on 5 Oct 2026,
when Diana ruled there is no need to code papers earlier than 2021. 2026 is a
partial year by construction (corpus snapshot 3 Sep 2026) — that is accepted,
not a defect to correct for. The stage machine already defaults to this window
(`BATCH_YEAR_MIN` in `v2/dap/pipeline.py`), so no flag is needed; the risk is
the opposite, passing `--year-min 2017` or `--oldest-first` by hand.
Two coders, blind, comparing on overlapping papers.
Every cited DOI is checked against DataCite — the reference list is not truncated.

**Open — do not assume an answer.**
- **853 already-classified papers are pre-2021**, coded before the window
  narrowed. The 24 Sep precedent is that nothing already coded is overwritten,
  so they stand — but they carry `das_present` 22.2% against 31.6% for
  2021–2026 (S213), so they move any pooled rate: the corpus is 2017–2026 in
  fact while coding is 2021–2026 in policy. Whether they stay is undecided.
  Until it is, report any rate **by year band, not pooled**. A further 4,770
  pre-2021 rows sit at `metadata/pending` and 699 at `retrieved/pending`; they
  are inert under the default window and need no cleanup.
- Which paper: the two-arm reproducibility-asymmetry comparison (quantum vs ML),
  or the single-population H1–H6 study of papers lacking a formal statement.
  Measured evidence favours the second: only 10.6% of the quantum corpus carries
  both axes, which makes the matched-pair design expensive and its sample
  unrepresentative; and 77% lack a formal statement, which makes H1–H6 cheap.
- The eight hard-case decision rules in the coding manual, unresolved between the
  two coders.
- Whether data reported only in a figure counts as available. The coding manual
  says no; hypothesis H2 assumes yes. These conflict and one must win.
- `necessity` and `access_route` variables, required by H3/H4, are **not yet in
  the schema**.
- The ML arm corpus does not exist. Only a quantum corpus has been collected.
- 2025 is under-indexed. Measured over all 543,258 parseable rows: 2023 peaks
  at 60,068, then 2024 55,995, then 2025 drops to 40,847. Partial 2026 already
  has 42,013 from a 3 Sep 2026 snapshot (~67% of the year), which annualises to
  roughly 62,000 — in line with the 2022→2023 trend. So 2026 behaves as a normal
  partial year and 2025 is the outlier, not 2026. The original note here said
  2025 was "about half of 2024"; it is 73%. Cause still unresolved.
  Also: exactly 1 row dated 2027, and 898 rows with an unparseable year.

## Hard rules

These exist because v1 violated each of them and the output had to be discarded.

1. **Never hand-classify papers.** If you are reading a paper and deciding its
   case yourself, stop — that is what produced v1's unusable output. Your job is
   to build and run the pipeline. The only hand-coding is done by the human
   coders, in the coding manual's terms.
2. **`schema.py` is the only source of truth for the vocabulary.** Never write a
   value that is not in its enums. If the validator raises, the input is wrong —
   fix the input, never catch-and-continue, never widen the enum to make an error
   go away.
3. **Never add, rename or re-mean a case token** without incrementing
   `SCHEME_VERSION` and asking first. v1 silently invented two codes and dropped
   three; that is why its data cannot be pooled.
4. **Never mix scheme versions across a vocabulary change.** Every row carries
   `scheme_version`, and it is an export column. The **patch** number records a
   ruling on how the existing tokens are applied (2.0.1: an explicit "no data"
   statement codes Axis B `not_applicable`; A is still coded from the paper.
   Diana, 28 Oct; current version). Rows differing only in patch
   pool on export, so a recode under a new ruling stays traceable and can be
   reversed if the ruling is. A change to the vocabulary itself (rule 3) bumps
   **major or minor**, and export refuses rows that differ there, on purpose —
   do not work around it. A ruling is a patch bump, never a quiet edit to rows
   still stamped with the old version. Changed from "export refuses any mix"
   on 28 Oct, at Diana's request.
5. **SQLite is the state.** CSV is an export. Never hand-edit either.
   One live row per DOI, enforced by a partial unique index (`ux_papers_live_doi`):
   OpenAlex mints more than one work id per DOI and `is_classified` keys on the
   id, so a paper enrolled twice is classified twice. Rows with no DOI and
   `excluded` tombstones sit outside the index on purpose. A database that
   already holds duplicates warns on open rather than refusing to open.
6. **Never claim verification that did not run.** `Actual` is written only by
   `verify.py`. `access_unresolvable` is derived from a failed check, never
   asserted from text. A check that did not *complete* is not a failed
   check: `Actual.ERROR` -- a connect timeout, a DNS failure, a 503 -- leaves the
   case alone. Only `not_found`, `resolves_no_data`, `paywalled` and
   `auth_required` promote, because only those establish that the path does not
   deliver. 19 verification rows carry `error` and 8 have no HTTP status at all.
7. **Flag, don't guess.** Low confidence sets `needs_human_review`. A flagged
   paper costs a human two minutes; a confident wrong code corrupts the study.
8. **Do not bypass bot detection, paywalls or CAPTCHAs.** Log the failure. The
   OA cascade exists precisely so this is unnecessary.
9. **Run the tests before saying something works.** `python -m pytest tests/ -q`,
   from `v2/` with the venv active (see Setup). 307 tests: `test_guards.py` 23,
   `test_integration.py` 172, `test_keyword_arms.py` 10, `test_pipeline.py` 83,
   `test_shard.py` 10, `test_handover.py` 9 —
   `test_pipeline.py` exercises the stage
   machine end-to-end against fakes, and calls `cmd_enrol` directly. Note what
   that split does *not* buy you: `test_integration.py` builds a `CrossrefFacts`
   by hand, so a feature can pass there and still be dead in the CLI because
   nothing carries its inputs across a stage boundary. That happened twice —
   `reference_count`, then `authors`. A retrieval feature needs a test in
   `test_pipeline.py` that starts from `stage_metadata_one`. The ~42%
   statement-coverage figure dates from 63 tests and has not been re-measured
   (`coverage` is not installed). Still untested: `main()`'s argparse wiring,
   `cmd_stage`/`cmd_status`/`cmd_export`, the `run_pool` thread-pool runner, and
   anything against the *live* internet — the fixtures encode a reading of the
   API docs, not recorded responses. A `--limit 25` shakedown is mandatory before
   any large run.
10. **Report real numbers.** Never estimate a count that can be measured. If you
    state a figure, it should come from a command you just ran.

## Layout

Every file in the repository, tracked or local, with its purpose and the reason it
was created or retired, is in `FILES.md`. The listing below covers only what a
session needs first.

```
FILES.md             the file map, measured state, uncommitted work, change log
v2/dap/schema.py     the coding scheme: enums, validation, tool schema
v2/dap/sources.py    Crossref/DataCite metadata first, full text only if needed
v2/dap/verify.py     does a declared access path actually resolve
v2/dap/classify.py   constrained LLM classification, rationale + confidence
v2/dap/store.py      SQLite state; CSV is an export
v2/dap/batch.py      Message Batches API at half price: spool/submit/collect
v2/dap/pipeline.py   staged, resumable, parallel runner — THE CLI
v2/tests/            307 offline tests incl. end-to-end stage machine; fakes.py fakes every boundary
v2/shard.py          lend a block of the queue to a coder on another machine; codes come
                     back through the same gate, checked against the live database first
v2/handover.py       move the state (database, scratch files, ledger, memory notes, PDFs,
                     harvest) to another machine and prove it arrived; HANDOVER.md is the guide
v2/agreement.py      inter-coder agreement, incl. the axis-inversion detector
v2/migrate.py        v1 rows -> v2 scheme
v2/migrated/         output of that migration
v2/review_issues.md  the running log: every case the pipeline could not settle,
                     every decision Diana has made, and the measurement behind it
v2/scratch/          MOSTLY ignored -- per-batch builders, prompt exports and
                     coded records are not in the repo. Eleven tools in it ARE
                     tracked, and .gitignore lists them by name:
  chk.py             MANDATORY before any `--import`: runs each record through
                     the real `_assemble` gate offline, grounding against the
                     passages in the DATABASE, because the store does not let a
                     quote come from the title/venue header the prompt carries.
                     Three rejections retire a paper for good.
  h.py               shared record builders, including one per decided policy
                     rule, so a batch cannot drift back to a pre-decision reading
  scan.py            one pass over a prompt file: statements (including
                     misspelled headings), identifiers, retrieval provenance
  triage.py          read-only: sorts every needs_human_review row into the nine
                     policy questions
  d2.py das.py v.py s.py b.py nostat.py
                     per-paper viewers, narrowest to widest
  spaced.py          read-only: GitHub addresses in stored passages that run on
                     past a space (a lost `_`), for rows coded before verify.py
                     learnt to repair them. A match is not a truncation: read it
../openalex_papers.csv  544,156 rows, quantum query, NOT filtered to topic
                     (one level ABOVE this repo, beside it — not inside)
```

## Setup

Use the existing venv, `.venv/` at this repo's root: Python 3.12, with everything
in `v2/requirements.txt` already installed. Run every command from `v2/`.

```bash
source .venv/bin/activate
cd v2
```

- **Never `pip install dap`.** `v2/dap/` is not an installable package (no
  `pyproject.toml` or `setup.py`); it runs as `python -m dap.pipeline` from `v2/`.
  `dap` on PyPI is an unrelated package, and installing it fails with
  `Failed to build 'dap' when getting requirements to build wheel`.
- **Not the system `python3`** (`/usr/bin/python3`, 3.9.6). Python 3.10+ is
  required; 3.9 fails on import with
  `dataclass() got an unexpected keyword argument 'slots'`.
- `DeprecationWarning: builtin type swigvarlink has no __module__ attribute` is
  harmless. It comes from PyMuPDF's SWIG bindings under Python 3.12, shows only
  under pytest (the last copy prints after the summary, at interpreter exit), and
  the installed 1.28.2 is already the latest release. No action needed.

## Commands

```bash
export DAP_CONTACT_EMAIL="you@institution.edu"   # required, real address
export ANTHROPIC_API_KEY="..."
# optional, once your library supplies them:
export ELSEVIER_TDM_KEY=... WILEY_TDM_TOKEN=... SPRINGER_TDM_KEY=...

python -m pytest tests/ -q
python -m dap.pipeline enrol --input ../../openalex_papers.csv --arm-map arms.csv \\
    --year-min 2021 --year-max 2026
python -m dap.pipeline stage metadata --workers 48          # no key needed
python -m dap.pipeline stage retrieve --workers 16          # no key needed
python -m dap.pipeline stage classify --mode api --limit 25 --budget 2.00   # shakedown FIRST
python -m dap.pipeline stage classify --mode batch          # census, half price
python -m dap.pipeline stage classify --mode collect
python -m dap.pipeline stage classify --mode manual --export p.jsonl   # agent-as-classifier, gated
python -m dap.pipeline stage verify   --workers 32          # no key needed
python -m dap.pipeline requeue W123 --reason "..."          # wrong copy stored; retrieved papers only
python -m dap.pipeline retry   W123 --reason "..."          # out of attempts at a stage; unclassified only
# removals keep the row and its codes; all report only, until --yes
python -m dap.pipeline exclude  W123 --reason "..." --yes   # duplicate, correction notice
python -m dap.pipeline outdated W123 --reason "..." --yes   # outside the study window
python -m dap.pipeline adjudicate W123 --set b_case=... --coder human:diana --reason "..."  # schema-gated; re-verifies a changed identifier
python -m dap.pipeline preprint-das W123                    # arXiv copy read in full: statement or not
# every `stage` takes the batch window: 2021+ and newest first by default
python -m dap.pipeline stage retrieve --year-min 2021        # 0 disables the window
python -m dap.pipeline stage retrieve --oldest-first         # reverse the order
python -m dap.pipeline stage classify --publisher-doi-only   # skip preprint-registrar DOIs
python -m dap.pipeline stage classify --skip-coded-dois      # one DOI, two OpenAlex ids
# 46% of the classify queue carried an arXiv/Zenodo/Research Square/SSRN DOI while
# 99.8% of what was already coded carried a publisher DOI, and 3,729 queued preprint
# rows duplicated a publisher row. `--publisher-doi-only` is a claim scope, not an
# exclusion: nothing is written and the rows return the moment it is off. Diana's
# call, 30 Sep 2026. The 1,504 Zenodo self-deposits were excluded outright.
python -m dap.pipeline status | export --out out/
python agreement.py coder_a.csv coder_b.csv --report agreement.md
```

Only `--mode api`/`batch` need `ANTHROPIC_API_KEY`. `--mode manual` runs the
agent as the classifier, through the identical `schema.py` gate — same
validation, same grounding check, no cost difference. It is not disqualified
from the census by being manual. Two things still apply regardless of mode:
  * **Pin the model.** `record_classification`'s `model_id` already tracks
    this (`manual:<coder>`). If the underlying agent model changes mid-run
    (it did once, Opus 5 -> Sonnet 5, mid-conversation), papers coded before
    and after are not from an identical protocol. Name the coder id after the
    specific model version, and do not mix versions within one coder's run.
  * **The blind two-coder comparison in Decided is a separate requirement.**
    It is not met by running manual mode once, API or not — there is still
    only one classifier. It needs an actual second, independent pass (human,
    or a second agent run blind to the first) over an overlapping sample, fed
    to `agreement.py`. This is still open.

## v1 — what it was and why it can't be reused directly

v1 defined **Axis A as the paper's own data** and Axis B as reused. The current
framework inverts this: **Axis A is external, Axis B is self-generated.** v1 also
used cases 1–10 with different meanings; old Case 1 was "full access" where new
case 1 is "not applicable" — near-opposites at the same number.

`migrate.py` transposes the axes and maps what it can. On the real 1,290 rows:
253 migrated cleanly, 1,037 need a human pass. The bulk of those are v1 case 7
(theory/simulation papers coded "no data necessary", which under v2 have a real
Axis B) and v1 case 5/6 (v2 needs identifiable-vs-too-vague; v1 never recorded it).

## Conventions

- Tokens, never labels, in stored data and analysis code.
- `CASE_NUMBER` is for publication; it is **not** monotone in openness. Use
  `OPENNESS_RANK` for anything ordinal.
- Report raw agreement, kappa and PABAK together. Kappa alone is misleading here:
  the case distribution is skewed, and 90% agreement can read as kappa 0.16.
- New dependencies: ask first.
- Do not commit API keys, `dap.sqlite`, or anything under `out/`.
- `FILES.md` changes in the same commit as anything that adds, removes, renames or
  re-purposes a file, or changes what a stage or command does. Update the file's
  entry, the change log, and the measured state if it moved. Where the reason a
  file exists is unknown, say so rather than inventing one.
